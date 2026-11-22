"""Metrics computed from a run's outcomes, the tickets' labels and the decision log.

Everything in the report is calculated here by code (A10). Figures that need
labels are computed when the input file carries them (the hidden set uses the same
schema) and are marked "n/a" otherwise. Figures that need human judgement
(hallucination rate, satisfaction) are not faked: the harness exports a review
sample for two assessors and reports automated proxies, clearly labelled.
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from src.guardrails import find_private_data
from src.schemas import Action, Outcome, Ticket

BASELINE = {"first_contact_resolution_pct": 42.0, "mean_first_reply": "8 to 12 hours", "csat": 3.2,
            "escalation_rate_pct": 58.0}
TARGETS = {"first_contact_resolution_pct": 60.0, "escalation_rate_pct": 30.0, "intent_precision": 0.85,
           "hallucination_rate_pct": 5.0, "citation_accuracy_pct": 95.0, "latency_p95_s": 3.0,
           "private_data_occurrences": 0, "cross_group_variation_pts": 5.0, "calibration_gap_pts": 5.0}


def _pct(n: float, d: float) -> float | None:
    return round(100.0 * n / d, 1) if d else None


def _pctl(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return round(xs[k], 1)


def _labels(t: Ticket) -> dict | None:
    lab = t.raw.get("labels") if isinstance(t.raw, dict) else None
    return lab if isinstance(lab, dict) and lab.get("intent") else None


def per_class(y_true: list[str], y_pred: list[str]) -> dict[str, dict]:
    classes = sorted(set(y_true) | set(y_pred))
    out = {}
    for c in classes:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"precision": round(p, 3), "recall": round(r, 3), "support": tp + fn}
    return out


def calibration(confs: list[float], correct: list[bool], bands: int = 5) -> list[dict]:
    rows = []
    for i in range(bands):
        lo, hi = i / bands, (i + 1) / bands
        idx = [k for k, c in enumerate(confs) if lo <= c < hi or (i == bands - 1 and c == 1.0)]
        if not idx:
            continue
        stated = sum(confs[k] for k in idx) / len(idx)
        observed = sum(correct[k] for k in idx) / len(idx)
        rows.append({"band": f"{lo:.1f}-{hi:.1f}", "n": len(idx), "stated": round(stated, 3),
                     "observed": round(observed, 3), "gap_pts": round(100 * (stated - observed), 1)})
    return rows


def compute(tickets: list[Ticket], outcomes: list[Outcome], log_finals: int, run_meta: dict) -> dict[str, Any]:
    n = len(outcomes)
    acts = Counter(o.action for o in outcomes)
    auto = [o for o in outcomes if o.action == Action.AUTO_RESPOND]
    blocked = [o for o in outcomes if o.action == Action.BLOCK]
    escalated_all = [o for o in outcomes if o.action != Action.AUTO_RESPOND]
    lat_s = [o.latency_ms / 1000 for o in outcomes]
    auto_lat = [o.latency_ms / 1000 for o in auto]

    # ---------------------------------------------------------------- volume
    volume = {"tickets_in_file": len(tickets), "tickets_processed": n, "answered_automatically": len(auto),
              "escalated": acts[Action.ESCALATE], "blocked_by_guardrails": len(blocked),
              "with_ingest_warnings": sum(1 for t in tickets if t.ingest_warnings),
              "recovered_errors": sum(1 for o in outcomes if any(not e.startswith("ingest:") for e in o.errors))}

    # ---------------------------------------------------------------- business
    business = {
        "first_contact_resolution_pct": _pct(len(auto), n),
        "first_contact_resolution_note": "share closed with no human involvement; assumes an auto-answer resolves "
                                         "the ticket, which only the satisfaction review and repeat contacts can confirm",
        "escalation_rate_pct": _pct(len(escalated_all), n),
        "time_to_first_reply_auto_s": {"mean": round(statistics.mean(auto_lat), 2) if auto_lat else None,
                                       "median": round(statistics.median(auto_lat), 2) if auto_lat else None,
                                       "p95": _pctl(auto_lat, 0.95)},
        "time_to_acknowledgement_all_s": {"mean": round(statistics.mean(lat_s), 2) if lat_s else None,
                                          "median": round(statistics.median(lat_s), 2) if lat_s else None},
        "first_reply_note": "escalated tickets receive an immediate acknowledgement; their substantive reply time "
                            "depends on agent availability and is not measured by this run",
        "baseline": BASELINE,
    }
    hist = [t.raw.get("history") for t in tickets if isinstance(t.raw.get("history"), dict)]
    if hist:
        business["historical_in_this_file"] = {
            "first_contact_resolution_pct": _pct(sum(1 for h in hist if h.get("first_contact_resolution")), len(hist)),
            "escalation_rate_pct": _pct(sum(1 for h in hist if h.get("escalated")), len(hist)),
            "mean_resolution_minutes": round(statistics.mean(h.get("resolution_time_minutes") or 0 for h in hist), 1),
            "median_resolution_minutes": statistics.median(h.get("resolution_time_minutes") or 0 for h in hist),
            "mean_csat": round(statistics.mean(h["csat_rating"] for h in hist if h.get("csat_rating")), 2),
            "repeat_contact_pct": _pct(sum(1 for h in hist if h.get("repeat_contact")), len(hist)),
        }

    # ---------------------------------------------------------------- technical
    technical: dict[str, Any] = {
        "latency_s": {"median": _pctl(lat_s, 0.5), "p95": _pctl(lat_s, 0.95), "max": _pctl(lat_s, 1.0)},
        "latency_note": "per-ticket processing time; time queued behind the free-tier rate limiter is excluded and "
                        "reported separately",
        "rate_limiter_wait_s_total": round(sum(o.throttle_ms for o in outcomes) / 1000, 1),
    }
    pairs = [(t, o) for t, o in zip(tickets, outcomes) if _labels(t)]
    if pairs:
        cls = [(t, o) for t, o in pairs if o.classification and not o.classification.fallback]
        yt = [_labels(t)["intent"] for t, _ in cls]
        yp = [o.classification.intent.value for _, o in cls]
        pc = per_class(yt, yp)
        technical["classification"] = {
            "n": len(cls),
            "intent_accuracy": round(sum(a == b for a, b in zip(yt, yp)) / len(yt), 3) if yt else None,
            "intent_macro_precision": round(statistics.mean(v["precision"] for v in pc.values()), 3) if pc else None,
            "intent_macro_recall": round(statistics.mean(v["recall"] for v in pc.values()), 3) if pc else None,
            "per_class": pc,
            "urgency_accuracy": round(sum(_labels(t)["urgency"] == o.classification.urgency.value
                                          for t, o in cls) / len(cls), 3) if cls else None,
            "calibration": calibration([o.classification.intent.confidence for _, o in cls],
                                       [a == b for a, b in zip(yt, yp)]),
        }
        # retrieval hit rate: expected article among retrieved, for tickets answerable from docs
        ans = [(t, o) for t, o in pairs if _labels(t).get("answerable_from_docs") and _labels(t).get("expected_doc_ids")]
        hits = sum(1 for t, o in ans if set(_labels(t)["expected_doc_ids"]) & {p.doc_id for p in o.passages})
        technical["retrieval_hit_rate_pct"] = _pct(hits, len(ans))
        unans = [(t, o) for t, o in pairs if not _labels(t).get("answerable_from_docs")]
        technical["unanswerable_auto_answered"] = sum(1 for _, o in unans if o.action == Action.AUTO_RESPOND)
        # citation accuracy (automated): cited article is one the labels say a correct answer should cite
        cites = [(t, c) for t, o in pairs if o.action == Action.AUTO_RESPOND and o.draft for c in o.draft.citations]
        ok = sum(1 for t, c in cites if c.doc_id in set(_labels(t).get("expected_doc_ids") or []))
        technical["citation_accuracy_pct"] = _pct(ok, len(cites))
        technical["citation_accuracy_note"] = ("automated: cited article matches the labelled expected articles; "
                                               "sentence-level support is checked by the grounding guardrail and the "
                                               "human review sample")
        # routing
        technical["routing"] = {
            "accuracy_pct": _pct(sum((o.action == Action.AUTO_RESPOND) == (_labels(t)["expected_route"] == "auto_respond")
                                     for t, o in pairs), len(pairs)),
            "auto_when_label_says_escalate": sum(1 for t, o in pairs if o.action == Action.AUTO_RESPOND
                                                 and _labels(t)["expected_route"] == "escalate"),
            "escalated_when_label_says_auto": sum(1 for t, o in pairs if o.action != Action.AUTO_RESPOND
                                                  and _labels(t)["expected_route"] == "auto_respond"),
        }
    technical["grounding_guardrail_block_rate_pct"] = _pct(
        sum(1 for o in outcomes if any(g.name == "grounding" and not g.passed for g in o.guardrails)),
        sum(1 for o in outcomes if o.draft and not o.draft.insufficient_evidence))
    technical["hallucination_rate_note"] = ("requires human review: see review_sample.csv (≥50 auto-answers, two "
                                            "assessors). The grounding block rate above is the automated proxy.")

    # ---------------------------------------------------------------- governance
    guard_counts = Counter(g.name for o in outcomes for g in o.guardrails if not g.passed)
    pii_out = [o.ticket_id for o in outcomes if find_private_data(o.response_text)]
    governance = {
        "decisions_logged_final": log_finals,
        "tickets_processed": n,
        "decision_log_reconciles": log_finals == n,
        "guardrail_activations_by_type": dict(guard_counts),
        "private_data_detections_in_outbound": len(pii_out),
        "private_data_detections_blocked_pre_send": guard_counts.get("private_data", 0),
        "must_not_auto_respond_violations": sum(1 for t, o in pairs if _labels(t).get("must_not_auto_respond")
                                                 and o.action == Action.AUTO_RESPOND) if pairs else "n/a",
    }
    if pairs:
        governance["fairness"] = fairness(pairs)
    return {"run": run_meta, "volume": volume, "business": business, "technical": technical,
            "governance": governance, "targets": TARGETS}


def fairness(pairs: list[tuple[Ticket, Outcome]]) -> dict:
    """Quality by customer group. 'Quality' = routing agreement with labels (did the ticket go where a senior
    agent says it should) and, for auto-answers, citation correctness. Gap = best minus worst segment."""
    seg_fns = {
        "customer_tier": lambda t: t.customer_tier,
        "language_fluency": lambda t: t.language_fluency,
        "customer_region": lambda t: t.customer_region,
        "channel": lambda t: t.channel.value,
        "length": lambda t: "short (<120 chars)" if len(t.body) < 120 else "long (>=120 chars)",
    }
    out = {}
    for seg, fn in seg_fns.items():
        groups: dict[str, list] = defaultdict(list)
        for t, o in pairs:
            groups[fn(t)].append((t, o))
        rows = {}
        for g, items in sorted(groups.items()):
            lab = [_labels(t) for t, _ in items]
            route_ok = sum((o.action == Action.AUTO_RESPOND) == (L["expected_route"] == "auto_respond")
                           for (t, o), L in zip(items, lab))
            should_auto = [(t, o) for t, o in items if _labels(t)["expected_route"] == "auto_respond"]
            cites = [(t, c) for t, o in items if o.action == Action.AUTO_RESPOND and o.draft for c in o.draft.citations]
            cite_ok = sum(1 for t, c in cites if c.doc_id in set(_labels(t).get("expected_doc_ids") or []))
            rows[g] = {"n": len(items), "routing_accuracy_pct": _pct(route_ok, len(items)),
                       "auto_rate_pct": _pct(sum(o.action == Action.AUTO_RESPOND for _, o in items), len(items)),
                       "resolution_rate_where_auto_expected_pct": _pct(
                           sum(o.action == Action.AUTO_RESPOND for _, o in should_auto), len(should_auto)),
                       "citation_accuracy_pct": _pct(cite_ok, len(cites))}
        vals = [r["resolution_rate_where_auto_expected_pct"] for r in rows.values()
                if r["resolution_rate_where_auto_expected_pct"] is not None and r["n"] >= 10]
        out[seg] = {"segments": rows,
                    "gap_pts_resolution_where_auto_expected": round(max(vals) - min(vals), 1) if len(vals) > 1 else None}
    return out
