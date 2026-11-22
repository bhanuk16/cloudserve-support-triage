"""Prometheus metrics. Exposed by the API at /metrics, and optionally by the harness
(--metrics-port) so a long unattended run can be watched live in Grafana.

What the dashboard shows (Setup Guide §06): tickets by channel and outcome, auto vs
escalated share, latency p50/p95, guardrail activations by type, and the confidence
distribution — the place drift shows up first.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from .schemas import Classification, Outcome

TICKETS = Counter("tickets_processed_total", "Tickets processed", ["channel", "outcome"])
LATENCY = Histogram("response_seconds", "End-to-end processing time per ticket",
                    buckets=(0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 8, 13, 21))
GUARDRAIL = Counter("guardrail_blocks_total", "Guardrail activations (blocking failures)", ["guardrail"])
CONFIDENCE = Histogram("intent_confidence", "Classifier intent confidence",
                       buckets=(0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0))
ERRORS = Counter("pipeline_errors_total", "Errors recovered from", ["component"])
KILL_SWITCH = Gauge("auto_respond_enabled", "1 when automatic answering is on, 0 when the kill switch is engaged")


def record_outcome(o: Outcome, c: Classification | None) -> None:
    TICKETS.labels(channel=o.channel, outcome=o.action.value).inc()
    LATENCY.observe(o.latency_ms / 1000)
    for g in o.guardrails:
        if not g.passed:
            GUARDRAIL.labels(guardrail=g.name).inc()
    if c and not c.fallback:
        CONFIDENCE.observe(c.intent.confidence)
    for e in o.errors:
        if not e.startswith("ingest:"):
            ERRORS.labels(component=e.split(":", 1)[0]).inc()
