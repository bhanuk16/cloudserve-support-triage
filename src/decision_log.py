"""Decision log: every automated decision, persisted with the fields the Governance
Framework requires (input summary, model + version, prediction + confidence,
alternatives, sources, threshold, action, plain-English reason, guardrail
results, prompt version, requirement ids).

Integrity:
  * append-only — SQLite triggers reject UPDATE and DELETE on the table;
  * tamper-evident — each record stores the SHA-256 of its own content plus the
    previous record's hash; `python -m src.decision_log --verify` re-walks the chain;
  * private data is not copied in: the input summary is the ticket text with
    emails, phone numbers, keys and card numbers masked.

Reconciliation (A8): each processed ticket produces exactly one stage='final'
record; `count_final()` must equal the number of tickets processed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    seq               INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id       TEXT NOT NULL UNIQUE,
    run_id            TEXT NOT NULL,
    timestamp         TEXT NOT NULL,
    ticket_id         TEXT NOT NULL,
    stage             TEXT NOT NULL CHECK (stage IN ('classification','retrieval','routing','generation','validation','final')),
    input_summary     TEXT,
    model             TEXT,
    prediction        TEXT,
    alternatives      TEXT,
    sources_used      TEXT,
    threshold_applied REAL,
    action_taken      TEXT,
    reason            TEXT,
    guardrail_results TEXT,
    prompt_version    TEXT,
    requirement_ids   TEXT,
    prev_hash         TEXT,
    record_hash       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_decisions_ticket ON decisions(ticket_id);
CREATE INDEX IF NOT EXISTS ix_decisions_run ON decisions(run_id, stage);
CREATE TRIGGER IF NOT EXISTS decisions_no_update BEFORE UPDATE ON decisions
BEGIN SELECT RAISE(ABORT, 'decision log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS decisions_no_delete BEFORE DELETE ON decisions
BEGIN SELECT RAISE(ABORT, 'decision log is append-only'); END;
"""

_MASKS = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[email]"),
    (re.compile(r"\b(sk|pk|rk|ghp|AKIA|AIza)[-_A-Za-z0-9]{12,}"), "[key]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[number]"),
    (re.compile(r"(?<![\w-])\+?\d[\d\s().-]{8,}\d(?![\w-])"), "[phone]"),
]
_FIELDS = ("decision_id", "run_id", "timestamp", "ticket_id", "stage", "input_summary", "model", "prediction",
           "alternatives", "sources_used", "threshold_applied", "action_taken", "reason", "guardrail_results",
           "prompt_version", "requirement_ids")


def mask(text: str, limit: int = 300) -> str:
    for rx, repl in _MASKS:
        text = rx.sub(repl, text)
    return text[:limit]


def _j(v: Any) -> str | None:
    return None if v is None else json.dumps(v, default=str, sort_keys=True)


class DecisionLog:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or settings.db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def write(self, *, run_id: str, ticket_id: str, stage: str, action_taken: str | None = None,
              reason: str = "", input_summary: str | None = None, model: dict | None = None,
              prediction: dict | None = None, alternatives: list | None = None, sources_used: list | None = None,
              threshold_applied: float | None = None, guardrail_results: dict | None = None,
              prompt_version: str | None = None, requirement_ids: list[str] | None = None) -> str:
        rec = {
            "decision_id": str(uuid.uuid4()), "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "ticket_id": ticket_id, "stage": stage,
            "input_summary": mask(input_summary) if input_summary else None,
            "model": _j(model), "prediction": _j(prediction), "alternatives": _j(alternatives),
            "sources_used": _j(sources_used), "threshold_applied": threshold_applied,
            "action_taken": action_taken, "reason": reason, "guardrail_results": _j(guardrail_results),
            "prompt_version": prompt_version, "requirement_ids": _j(requirement_ids or []),
        }
        with self._lock:
            row = self._conn.execute("SELECT record_hash FROM decisions ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else "GENESIS"
            digest = hashlib.sha256((prev + json.dumps([rec[f] for f in _FIELDS], default=str)).encode()).hexdigest()
            self._conn.execute(
                f"INSERT INTO decisions ({', '.join(_FIELDS)}, prev_hash, record_hash) "
                f"VALUES ({', '.join('?' * (len(_FIELDS) + 2))})",
                [rec[f] for f in _FIELDS] + [prev, digest],
            )
        return rec["decision_id"]

    # -- audit ---------------------------------------------------------------
    def count_final(self, run_id: str) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM decisions WHERE run_id=? AND stage='final'",
                                  (run_id,)).fetchone()[0]

    def finals(self, run_id: str) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM decisions WHERE run_id=? AND stage='final' ORDER BY seq", (run_id,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def for_ticket(self, ticket_id: str) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM decisions WHERE ticket_id=? ORDER BY seq", (ticket_id,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def verify_chain(self) -> tuple[bool, int, str]:
        prev, n = "GENESIS", 0
        cur = self._conn.execute(f"SELECT {', '.join(_FIELDS)}, prev_hash, record_hash FROM decisions ORDER BY seq")
        for row in cur:
            n += 1
            rec, prev_hash, rec_hash = row[:-2], row[-2], row[-1]
            expect = hashlib.sha256((prev + json.dumps(list(rec), default=str)).encode()).hexdigest()
            if prev_hash != prev or rec_hash != expect:
                return False, n, f"chain broken at record {n} (decision {rec[0]})"
            prev = rec_hash
        return True, n, "ok"

    def close(self) -> None:
        self._conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Decision log audit tools")
    ap.add_argument("--verify", action="store_true", help="verify the tamper-evident hash chain")
    ap.add_argument("--ticket", help="print every decision recorded for one ticket")
    a = ap.parse_args()
    dl = DecisionLog()
    if a.verify:
        ok, n, msg = dl.verify_chain()
        print(f"{'OK' if ok else 'FAILED'}: {n} records checked — {msg}")
        raise SystemExit(0 if ok else 1)
    if a.ticket:
        for r in dl.for_ticket(a.ticket):
            print(json.dumps(r, indent=2))
