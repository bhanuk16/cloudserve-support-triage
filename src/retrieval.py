"""Retrieve: find the documentation passages that bear on a ticket.

Chunking decision (see docs/design-decisions.md, DD-02):
  Every article has the same four sections — Symptoms, Common causes, Resolution,
  Notes — and is only ~1,000 characters long. We index each *section* as its own
  passage, prefixed with the article title, so:
    * a customer's description ("my deployment keeps dying") matches the Symptoms
      section, which is written in problem language, not the solution-language title;
    * a citation points at a specific section (e.g. DOC-DEPLOY-001#resolution), which
      is precise enough to verify (A6), and never splits a numbered resolution
      sequence half-way (the Dataset Guide warns about exactly this).
  At answer time we expand matched sections back to their whole article
  (small-to-big), because a correct answer usually needs Symptoms *and* Resolution.

Scoring: dense cosine similarity (all-MiniLM-L6-v2) is the relevance gate because
it is an absolute, comparable number; BM25 adds a bounded boost for exact terms
such as error codes, "401", "SAML" or CLI commands that embeddings blur. Below the
threshold we return nothing — an empty result is a valid answer (Build Spec).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings
from .embeddings import Embedder, get_embedder
from .schemas import Passage

_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the and or of to in on for is are was were be been it its this that with as at by from "
    "i we our you your my me us they them can could would should will do does did not no have has "
    "had how what why when where which who please any some there here so if but about into than".split()
)
BM25_WEIGHT = 0.15  # boost ceiling; dense similarity stays the dominant, comparable signal


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass(frozen=True)
class Chunk:
    passage_id: str
    doc_id: str
    title: str
    section: str
    text: str  # section body only — what gets cited and shown

    @property
    def index_text(self) -> str:  # what gets embedded: title gives the section context
        return f"{self.title}. {self.section}: {self.text}"


def load_corpus(path: Path | None = None) -> list[dict]:
    return json.loads(Path(path or settings.docs_path).read_text(encoding="utf-8"))


def chunk_document(doc: dict) -> list[Chunk]:
    content = doc.get("content", "")
    title = doc.get("title", doc["doc_id"])
    parts = _SECTION.split(content)
    chunks: list[Chunk] = []
    # parts = [preamble, heading1, body1, heading2, body2, ...]
    for heading, body in zip(parts[1::2], parts[2::2]):
        body = body.strip()
        if not body:
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")
        chunks.append(Chunk(f"{doc['doc_id']}#{slug}", doc["doc_id"], title, heading.strip(), body))
    if not chunks:  # an article without the usual headings is indexed whole
        chunks.append(Chunk(f"{doc['doc_id']}#full", doc["doc_id"], title, "Full article", content.strip()))
    return chunks


class Retriever:
    """Hybrid section retriever backed by a persistent Chroma collection."""

    COLLECTION = "cloudserve_docs"

    def __init__(self, embedder: Embedder | None = None, docs_path: Path | None = None,
                 chroma_path: Path | None = None, min_score: float | None = None):
        self.embedder = embedder or get_embedder()
        self.docs = {d["doc_id"]: d for d in load_corpus(docs_path)}
        self.chunks = [c for d in self.docs.values() for c in chunk_document(d)]
        self.by_id = {c.passage_id: c for c in self.chunks}
        self.min_score = settings.retrieval_min_score if min_score is None else min_score
        self._bm25 = BM25Okapi([tokenize(c.index_text) for c in self.chunks])
        self._lock = threading.Lock()
        self._matrix = self._load_or_build_index(Path(chroma_path or settings.chroma_path))

    # -- index ---------------------------------------------------------------
    def _corpus_fingerprint(self) -> str:
        h = hashlib.sha256()
        for c in self.chunks:
            h.update(c.passage_id.encode()); h.update(c.index_text.encode())
        return h.hexdigest()[:16]

    def _load_or_build_index(self, chroma_path: Path) -> np.ndarray:
        """Chroma persists the vectors; a corpus change (new fingerprint) forces a rebuild,
        so the index can never silently drift from the documentation (risk R-05)."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_path), settings=ChromaSettings(anonymized_telemetry=False))
        fingerprint = self._corpus_fingerprint()
        name = f"{self.COLLECTION}_{fingerprint}"
        existing = {c.name for c in client.list_collections()}
        for stale in existing - {name}:
            if stale.startswith(self.COLLECTION):
                client.delete_collection(stale)
        col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"}, embedding_function=None)
        ids = [c.passage_id for c in self.chunks]
        if col.count() != len(ids):
            vectors = self.embedder.embed([c.index_text for c in self.chunks])
            col.upsert(ids=ids, embeddings=vectors.tolist(), documents=[c.text for c in self.chunks],
                       metadatas=[{"doc_id": c.doc_id, "section": c.section, "title": c.title} for c in self.chunks])
        got = col.get(ids=ids, include=["embeddings"])
        order = {pid: i for i, pid in enumerate(got["ids"])}
        # keep an in-memory copy: exact search over ~120 vectors is faster than ANN and fully deterministic
        return np.asarray([got["embeddings"][order[pid]] for pid in ids], dtype=np.float32)

    # -- search --------------------------------------------------------------
    def score_all(self, query: str) -> np.ndarray:
        q = self.embedder.embed_one(query)
        dense = self._matrix @ q
        bm = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
        bm = bm / bm.max() if bm.max() > 0 else bm
        return dense + BM25_WEIGHT * bm * np.clip(dense, 0, None)  # boost only what is already relevant

    def search(self, query: str, top_k: int | None = None, min_score: float | None = None) -> list[Passage]:
        top_k = top_k or settings.retrieval_top_k
        floor = self.min_score if min_score is None else min_score
        if not query or not query.strip():
            return []
        scores = self.score_all(query)
        order = np.argsort(-scores, kind="stable")
        out: list[Passage] = []
        for i in order[:top_k]:
            if scores[i] < floor:
                break
            c = self.chunks[i]
            out.append(Passage(passage_id=c.passage_id, doc_id=c.doc_id, title=c.title,
                               section=c.section, text=c.text, score=round(float(scores[i]), 4)))
        return out

    def expand_to_documents(self, passages: list[Passage], max_docs: int = 2, margin: float = 0.05) -> list[Passage]:
        """Small-to-big: return every section of the best-matching articles, in article order,
        each carrying the score of the best hit in its article. A second article is included only
        when it scores within `margin` of the best, so a weaker article cannot dilute the context."""
        best: dict[str, float] = {}
        for p in passages:
            best[p.doc_id] = max(best.get(p.doc_id, 0.0), p.score)
        ranked = sorted(best, key=lambda d: -best[d])
        top_docs = [d for d in ranked[:max_docs] if best[d] >= best[ranked[0]] - margin]
        out = []
        for doc_id in top_docs:
            for c in chunk_document(self.docs[doc_id]):
                out.append(Passage(passage_id=c.passage_id, doc_id=c.doc_id, title=c.title,
                                   section=c.section, text=c.text, score=best[doc_id]))
        return out

    def resolve(self, passage_id: str) -> Chunk | None:
        """Citation check helper: does this identifier exist in the real corpus?"""
        return self.by_id.get(passage_id)


_retriever: Retriever | None = None
_rlock = threading.Lock()


def get_retriever() -> Retriever:
    global _retriever
    with _rlock:
        if _retriever is None:
            _retriever = Retriever()
        return _retriever
