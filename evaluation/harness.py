"""Unattended evaluation run (A9, A10).

    python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/

Processes every ticket in the input file, never stops on a bad ticket, writes each
outcome as it goes, then produces the metrics report — with no manual steps.

Outputs (in --output):
  results.jsonl         one line per ticket: action, reason, response, citations, guardrails, latency
  metrics.json          every figure the Build Specification requires, computed by code
  metrics_report.md     the same, readable
  review_sample.csv     ≥50 auto-answers for two human assessors (hallucination rate, satisfaction proxy)
  escalations.csv       the handover each escalated ticket carried
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings

log = logging.getLogger("harness")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=Path(__file__).parent, timeout=5).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def preflight() -> None:
    """Make a clean checkout self-healing: build what is missing, fail loudly only if it cannot."""
    from src.embeddings import ModelNotInstalled, get_embedder

    try:
        get_embedder()
    except ModelNotInstalled:
        from src import setup_models
        if setup_models.main([]) != 0:
            raise
    if not (settings.artifacts_dir / "classifier.npz").exists():
        log.info("classifier not trained yet; training from %s", settings.train_path)
        from src.classify import train
        train()


def run(input_path: Path, output_dir: Path, limit: int | None = None, metrics_port: int | None = None) -> dict:
    from evaluation.metrics import compute
    from src.ingest import load_tickets
    from src.pipeline import Pipeline

    output_dir.mkdir(parents=True, exist_ok=True)
    preflight()
    if metrics_port:
        from prometheus_client import start_http_server
        start_http_server(metrics_port)

    tickets = load_tickets(input_path)
    if limit:
        tickets = tickets[:limit]
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    pipe = Pipeline()
    meta = {
        "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_file": str(input_path), "input_sha256_12": hashlib.sha256(input_path.read_bytes()).hexdigest()[:12],
        "system_commit": _git_commit(), "llm_provider": settings.llm_provider, "llm_model": settings.model_name,
        "classifier_version": pipe.classifier.version, "routing_policy": pipe.router.version,
        "kill_switch_engaged": pipe.router.kill_switch_on(),
    }
    log.info("run %s: %d tickets from %s", run_id, len(tickets), input_path)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))  # finish the current ticket, then report

    outcomes = []
    t0 = time.perf_counter()
    with open(output_dir / "results.jsonl", "w", encoding="utf-8") as fh:
        for i, ticket in enumerate(tickets, 1):
            if stop["flag"]:
                log.warning("interrupted after %d tickets; writing a partial report", i - 1)
                break
            o = pipe.process(ticket, run_id=run_id)  # never raises
            outcomes.append(o)
            fh.write(json.dumps({
                "ticket_id": o.ticket_id, "channel": o.channel, "action": o.action.value, "reason": o.reason,
                "intent": o.classification.intent.value if o.classification else None,
                "intent_confidence": o.classification.intent.confidence if o.classification else None,
                "urgency": o.classification.urgency.value if o.classification else None,
                "sources": [p.passage_id for p in o.passages],
                "citations": [c.passage_id for c in o.draft.citations] if o.draft else [],
                "guardrails": {g.name: g.passed for g in o.guardrails},
                "latency_ms": o.latency_ms, "errors": o.errors,
                "response": o.response_text, "escalation_summary": o.escalation_summary,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(tickets):
                rate = i / (time.perf_counter() - t0)
                print(f"  {i}/{len(tickets)} processed ({rate:.1f}/s)", file=sys.stderr, flush=True)

    meta["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta["wall_clock_s"] = round(time.perf_counter() - t0, 1)
    meta["llm_circuit_breaker"] = pipe.llm.breaker.state if hasattr(pipe.llm, "breaker") else "n/a"
    report = compute(tickets[: len(outcomes)], outcomes, pipe.dlog.count_final(run_id), meta)
    ok, n_rec, msg = pipe.dlog.verify_chain()
    report["governance"]["decision_log_integrity"] = {"hash_chain_ok": ok, "records_checked": n_rec, "detail": msg}

    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (output_dir / "metrics_report.md").write_text(render_markdown(report), encoding="utf-8")
    _write_review_sample(output_dir / "review_sample.csv", outcomes, tickets)
    _write_escalations(output_dir / "escalations.csv", outcomes)
    return report


def _write_review_sample(path: Path, outcomes, tickets) -> None:
    by_id = {t.ticket_id: t for t in tickets}
    auto = [o for o in outcomes if o.action.value == "auto_respond"]
    sample = random.Random(42).sample(auto, min(len(auto), 60))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticket_id", "ticket_text", "response", "cited_passages",
                    "A1_unsupported_claim(0/1)", "A2_unsupported_claim(0/1)", "A1_quality(1-5)", "A2_quality(1-5)", "notes"])
        for o in sample:
            w.writerow([o.ticket_id, by_id[o.ticket_id].text, o.response_text,
                        " | ".join(f"{p.passage_id}: {p.text}" for p in o.passages[:3]), "", "", "", "", ""])


def _write_escalations(path: Path, outcomes) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticket_id", "action", "reason", "handover_summary"])
        for o in outcomes:
            if o.action.value != "auto_respond":
                w.writerow([o.ticket_id, o.action.value, o.reason, o.escalation_summary])


def render_markdown(r: dict) -> str:
    v, b, t, g = r["volume"], r["business"], r["technical"], r["governance"]
    L = [f"# Evaluation report — {r['run']['run_id']}", "",
         f"Input `{r['run']['input_file']}` (sha {r['run']['input_sha256_12']}), system commit "
         f"`{r['run']['system_commit']}`, model `{r['run']['llm_model']}` via {r['run']['llm_provider']}, "
         f"classifier `{r['run']['classifier_version']}`, policy {r['run']['routing_policy']}. "
         f"Wall clock {r['run']['wall_clock_s']} s.", "",
         "## Volume", "| Measure | Value |", "|---|---|"]
    L += [f"| {k.replace('_', ' ')} | {val} |" for k, val in v.items()]
    L += ["", "## Business outcomes", "| Measure | Achieved | Baseline | Target |", "|---|---|---|---|",
          f"| First contact resolution | {b['first_contact_resolution_pct']}% | 42% | ≥60% |",
          f"| Escalation rate | {b['escalation_rate_pct']}% | 58% | ≤30% |",
          f"| Time to first reply (auto-answered), mean / median | {b['time_to_first_reply_auto_s']['mean']} s / "
          f"{b['time_to_first_reply_auto_s']['median']} s | 8–12 h | <5 min |", "",
          f"_{b['first_contact_resolution_note']}. {b['first_reply_note']}._", ""]
    if "historical_in_this_file" in b:
        L += ["Historical (human-handled) figures for the same tickets: " +
              ", ".join(f"{k.replace('_', ' ')} {val}" for k, val in b["historical_in_this_file"].items()), ""]
    L += ["## Technical", "| Measure | Value | Target |", "|---|---|---|",
          f"| Latency median / p95 | {t['latency_s']['median']} s / {t['latency_s']['p95']} s | p95 < 3 s |"]
    if "classification" in t:
        c = t["classification"]
        L += [f"| Intent accuracy | {c['intent_accuracy']} | — |",
              f"| Intent macro precision / recall | {c['intent_macro_precision']} / {c['intent_macro_recall']} | ≥0.85 |",
              f"| Urgency accuracy | {c['urgency_accuracy']} | — |",
              f"| Retrieval hit rate | {t['retrieval_hit_rate_pct']}% | — |",
              f"| Citation accuracy (automated) | {t['citation_accuracy_pct']}% | ≥95% |",
              f"| Routing agreement with labels | {t['routing']['accuracy_pct']}% | — |",
              f"| Grounding guardrail block rate | {t['grounding_guardrail_block_rate_pct']}% | — |"]
        L += ["", "### Calibration (intent)", "| Band | n | Stated | Observed | Gap (pts) |", "|---|---|---|---|---|"]
        L += [f"| {x['band']} | {x['n']} | {x['stated']} | {x['observed']} | {x['gap_pts']} |" for x in c["calibration"]]
        L += ["", "### Per-class precision / recall", "| Intent | Precision | Recall | Support |", "|---|---|---|---|"]
        L += [f"| {k} | {x['precision']} | {x['recall']} | {x['support']} |" for k, x in c["per_class"].items()]
    L += ["", f"_{t['hallucination_rate_note']}_", "", "## Governance", "| Condition | Value |", "|---|---|",
          f"| Decisions logged (final) vs tickets processed | {g['decisions_logged_final']} / {g['tickets_processed']} "
          f"({'reconciles' if g['decision_log_reconciles'] else 'DOES NOT RECONCILE'}) |",
          f"| Decision log hash chain | {'intact' if g['decision_log_integrity']['hash_chain_ok'] else 'BROKEN'} |",
          f"| Private data in outbound responses | {g['private_data_detections_in_outbound']} |",
          f"| Private data caught before sending | {g['private_data_detections_blocked_pre_send']} |",
          f"| Auto-answers on never-auto ticket types | {g['must_not_auto_respond_violations']} |",
          f"| Guardrail activations by type | {g['guardrail_activations_by_type'] or 'none'} |"]
    if "fairness" in g:
        L += ["", "### Fairness — resolution rate where a senior agent says auto-answer is right"]
        for seg, d in g["fairness"].items():
            L += [f"**{seg}** (gap {d['gap_pts_resolution_where_auto_expected']} pts)", "",
                  "| Segment | n | Routing accuracy | Auto rate | Resolution where auto expected | Citation accuracy |",
                  "|---|---|---|---|---|---|"]
            L += [f"| {k} | {x['n']} | {x['routing_accuracy_pct']}% | {x['auto_rate_pct']}% | "
                  f"{x['resolution_rate_where_auto_expected_pct']}% | {x['citation_accuracy_pct']}% |"
                  for k, x in d["segments"].items()]
            L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Process a ticket file unattended and produce the metrics report.")
    ap.add_argument("--input", required=True, type=Path, help="ticket file (JSON list, same schema as the pack)")
    ap.add_argument("--output", required=True, type=Path, help="directory for results and the metrics report")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N tickets (smoke tests)")
    ap.add_argument("--metrics-port", type=int, default=None, help="expose Prometheus metrics while running")
    a = ap.parse_args(argv)
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not a.input.exists():
        print(f"input file not found: {a.input}", file=sys.stderr)
        return 2
    r = run(a.input, a.output, a.limit, a.metrics_port)
    v, g = r["volume"], r["governance"]
    print(f"\nDone: {v['tickets_processed']} processed — {v['answered_automatically']} answered, "
          f"{v['escalated']} escalated, {v['blocked_by_guardrails']} blocked. "
          f"Decision log {'reconciles' if g['decision_log_reconciles'] else 'DOES NOT reconcile'}. "
          f"Report: {a.output / 'metrics_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
