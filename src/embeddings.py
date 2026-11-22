"""all-MiniLM-L6-v2 sentence embeddings, run locally through ONNX Runtime.

Why ONNX rather than sentence-transformers: it is the same model the brief
specifies, but it needs no PyTorch install (hundreds of MB, fragile on Windows),
it starts in well under a second, and it never calls the network at run time.
The model files are fetched once, checksum-verified, by `python -m src.setup_models`.

Pooling and normalisation follow sentence-transformers exactly: attention-masked
mean pooling over the last hidden state, then L2 normalisation, so dot product
equals cosine similarity.
"""
from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import settings

MAX_TOKENS = 256  # sentence-transformers' max_seq_length for this model


class ModelNotInstalled(RuntimeError):
    pass


class Embedder:
    def __init__(self, model_dir: Path | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir or settings.model_dir)
        onnx_path, tok_path = model_dir / "model.onnx", model_dir / "tokenizer.json"
        if not onnx_path.exists() or not tok_path.exists():
            raise ModelNotInstalled(
                f"Embedding model not found in {model_dir}. Run: python -m src.setup_models"
            )
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")  # pad to longest in batch
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 0  # let ORT choose
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self.session.get_inputs()}
        self._lock = threading.Lock()  # tokenizer padding state is not thread-safe

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [t if t.strip() else "[empty]" for t in texts[i : i + batch_size]]
            with self._lock:
                enc = self.tokenizer.encode_batch(batch)
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feeds)[0]
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            out.append((pooled / np.clip(norm, 1e-12, None)).astype(np.float32))
        return np.concatenate(out)

    def embed_one(self, text: str) -> np.ndarray:
        return _cached_query(self, text)


@lru_cache(maxsize=4096)
def _cached_query(embedder: "Embedder", text: str) -> np.ndarray:
    return embedder.embed([text])[0]


_instance: Embedder | None = None
_instance_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = Embedder()
        return _instance
