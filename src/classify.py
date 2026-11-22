"""Classify: intent, urgency and answerability, each with a calibrated confidence.

Design (docs/design-decisions.md, DD-03):
  * A local model, not the LLM. Multinomial logistic regression over
    all-MiniLM-L6-v2 sentence embeddings. It is deterministic (A5), runs in
    milliseconds, costs nothing, and keeps working when the model provider is down.
  * Confidence is calibrated with temperature scaling fitted on out-of-fold
    predictions (GroupKFold by ticket text, so duplicated wording never leaks
    between folds). A calibrated 0.85 means "right about 85% of the time", which is
    what makes a routing threshold meaningful (governance: within 5 points).
  * The saved model is plain arrays in a .npz loaded with allow_pickle=False plus a
    JSON manifest — no pickle, so a tampered artefact cannot execute code on load.

Train:  python -m src.classify --train      (about 10 seconds; deterministic)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import settings
from .embeddings import get_embedder
from .schemas import Classification, Scored, Ticket

log = logging.getLogger(__name__)

FALLBACK_INTENT = "unclear_request"  # routes to a human by policy
FALLBACK_URGENCY = "medium"
HEADS = ("intent", "urgency", "answerable")
SEED = 42


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class Head:
    classes: list[str]
    W: np.ndarray  # (n_classes, dim)
    b: np.ndarray  # (n_classes,)
    T: float  # temperature

    def proba(self, X: np.ndarray) -> np.ndarray:
        return _softmax((X @ self.W.T + self.b) / self.T)


class Classifier:
    def __init__(self, artifact_dir: Path | None = None):
        d = Path(artifact_dir or settings.artifacts_dir)
        manifest = json.loads((d / "classifier.json").read_text())
        arrays = np.load(d / "classifier.npz", allow_pickle=False)
        self.version = manifest["version"]
        self.manifest = manifest
        self.heads = {
            h: Head(manifest["heads"][h]["classes"], arrays[f"{h}_W"], arrays[f"{h}_b"],
                    float(manifest["heads"][h]["temperature"]))
            for h in HEADS
        }
        self.embedder = get_embedder()

    def predict(self, ticket: Ticket) -> Classification:
        try:
            x = self.embedder.embed_one(ticket.text or "")[None, :]
            out = {}
            for name, head in self.heads.items():
                p = head.proba(x)[0]
                order = np.argsort(-p, kind="stable")
                out[name] = [Scored(value=head.classes[i], confidence=round(float(p[i]), 4)) for i in order]
            answerable = next(s for s in out["answerable"] if s.value == "true")
            return Classification(
                intent=out["intent"][0],
                urgency=out["urgency"][0],
                answerable=Scored(value="true", confidence=answerable.confidence),
                intent_alternatives=out["intent"][1:4],
                model_version=self.version,
            )
        except Exception as exc:  # never raise: a defined fallback that routes to a human
            log.exception("classifier failed for %s", ticket.ticket_id)
            return fallback(str(exc))


def fallback(reason: str = "") -> Classification:
    return Classification(
        intent=Scored(value=FALLBACK_INTENT, confidence=0.0),
        urgency=Scored(value=FALLBACK_URGENCY, confidence=0.0),
        answerable=Scored(value="true", confidence=0.0),
        model_version=f"fallback:{reason[:80]}",
        fallback=True,
    )


# --------------------------------------------------------------------------- training
def _fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float:
    """Pick the temperature that minimises out-of-fold negative log-likelihood."""
    best_t, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.05), np.log(10), 400)):
        p = _softmax(logits / t)
        nll = -np.mean(np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-12, None)))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def labels_for(tickets: list[Ticket]) -> dict[str, list[str]]:
    return {
        "intent": [t.raw["labels"]["intent"] for t in tickets],
        "urgency": [t.raw["labels"]["urgency"] for t in tickets],
        "answerable": [str(bool(t.raw["labels"]["answerable_from_docs"])).lower() for t in tickets],
    }


C_BY_HEAD = {"intent": 16.0, "urgency": 1.0, "answerable": 4.0}  # chosen by grouped CV, see docs


def train(train_path: Path | None = None, out_dir: Path | None = None, folds: int = 5) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import GroupKFold

    from .ingest import load_tickets

    train_path = Path(train_path or settings.train_path)
    out_dir = Path(out_dir or settings.artifacts_dir)
    tickets = [t for t in load_tickets(train_path) if t.raw.get("labels")]
    X = get_embedder().embed([t.text for t in tickets])
    groups = [t.body for t in tickets]  # identical wording never straddles folds
    labels = labels_for(tickets)
    arrays, manifest_heads, report = {}, {}, {}
    oof_rows = [{"ticket_id": t.ticket_id} for t in tickets]

    for head in HEADS:
        y = np.array(labels[head])
        classes = sorted(set(y))
        y_idx = np.array([classes.index(v) for v in y])
        oof_logits = np.zeros((len(y), len(classes)))
        for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
            m = LogisticRegression(C=C_BY_HEAD[head], max_iter=5000, random_state=SEED)
            m.fit(X[tr], y_idx[tr])
            full = np.full((len(te), len(classes)), -1e9)
            full[:, m.classes_] = m.decision_function(X[te]) if len(m.classes_) > 2 else np.c_[
                -m.decision_function(X[te]) / 2, m.decision_function(X[te]) / 2]
            oof_logits[te] = full
        T = _fit_temperature(oof_logits, y_idx)
        oof_p = _softmax(oof_logits / T)
        pred = oof_p.argmax(1)
        conf = oof_p.max(1)
        for i, row in enumerate(oof_rows):
            row[f"{head}_true"] = y[i]
            row[f"{head}_pred"] = classes[pred[i]]
            row[f"{head}_conf"] = round(float(conf[i]), 4)
            if head == "answerable":
                row["p_answerable"] = round(float(oof_p[i, classes.index("true")]), 4)
        report[head] = {
            "cv_accuracy": round(accuracy_score(y_idx, pred), 4),
            "cv_macro_precision": round(precision_score(y_idx, pred, average="macro", zero_division=0), 4),
            "cv_macro_recall": round(recall_score(y_idx, pred, average="macro", zero_division=0), 4),
            "cv_macro_f1": round(f1_score(y_idx, pred, average="macro", zero_division=0), 4),
            "temperature": round(T, 4),
            "calibration": calibration_table(conf, pred == y_idx),
        }
        final = LogisticRegression(C=C_BY_HEAD[head], max_iter=5000, random_state=SEED).fit(X, y_idx)
        if len(classes) == 2:  # store binary as two-row softmax so inference is uniform
            w, b0 = final.coef_[0], final.intercept_[0]
            W, b = np.stack([-w / 2, w / 2]), np.array([-b0 / 2, b0 / 2])
        else:
            W, b = final.coef_, final.intercept_
        arrays[f"{head}_W"], arrays[f"{head}_b"] = W.astype(np.float32), b.astype(np.float32)
        manifest_heads[head] = {"classes": classes, "temperature": T, "C": C_BY_HEAD[head]}

    data_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()[:12]
    manifest = {
        "version": f"logreg-minilm-{data_hash}",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_file": train_path.name,
        "train_sha256_12": data_hash,
        "n_train": len(tickets),
        "seed": SEED,
        "heads": manifest_heads,
        "cv_report": report,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "classifier.npz", **arrays)
    (out_dir / "classifier.json").write_text(json.dumps(manifest, indent=2))
    # out-of-fold predictions: the honest basis for choosing routing thresholds
    (out_dir / "oof_predictions.json").write_text(json.dumps(oof_rows, indent=1))
    return manifest


def calibration_table(confidence: np.ndarray, correct: np.ndarray, bands: int = 5) -> list[dict]:
    rows = []
    for i in range(bands):
        lo, hi = i / bands, (i + 1) / bands
        mask = (confidence >= lo) & ((confidence < hi) if i < bands - 1 else (confidence <= hi))
        if mask.sum() == 0:
            continue
        stated, observed = float(confidence[mask].mean()), float(correct[mask].mean())
        rows.append({"band": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()), "stated": round(stated, 3),
                     "observed": round(observed, 3), "gap": round(stated - observed, 3)})
    return rows


_clf: Classifier | None = None


def get_classifier() -> Classifier:
    global _clf
    if _clf is None:
        _clf = Classifier()
    return _clf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--input", type=Path, default=None, help="labelled ticket file to train on")
    a = ap.parse_args()
    if a.train:
        m = train(a.input)
        print(json.dumps({"version": m["version"], "cv_report": m["cv_report"]}, indent=2))
