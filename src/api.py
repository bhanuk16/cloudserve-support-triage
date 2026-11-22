"""HTTP interface (FastAPI). Interactive docs at http://127.0.0.1:8000/docs.

    python -m src.api

Endpoints
  POST /tickets            process one ticket from any channel; returns the decision
  GET  /decisions/{id}     the full decision trail for a ticket (audit)
  GET  /health             liveness + readiness (models loaded, kill switch state)
  GET  /metrics            Prometheus metrics
  POST /admin/kill-switch  engage/release the kill switch (needs ADMIN_TOKEN)

Security: binds to 127.0.0.1 by default; request bodies are size-limited and
schema-validated; the admin endpoint requires a bearer token compared in constant
time and is disabled entirely when ADMIN_TOKEN is unset; responses never echo
internal exception text.
"""
from __future__ import annotations

import hmac
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from .config import settings
from .ingest import normalise
from .monitoring import KILL_SWITCH

log = logging.getLogger("api")
MAX_REQUEST_BYTES = 64 * 1024
_state: dict = {}
_lock = threading.Lock()


class TicketIn(BaseModel):
    ticket_id: str = Field(..., max_length=64)
    channel: Literal["email", "chat", "docs_comment", "forum"]
    subject: str = Field("", max_length=300)
    body: str = Field(..., max_length=20000)
    received_at: str | None = None
    customer_id: str = Field("", max_length=64)
    customer_name: str = Field("", max_length=120)
    customer_tier: str = Field("standard", max_length=30)
    customer_region: str = Field("unknown", max_length=30)
    language_fluency: str = Field("fluent", max_length=30)


class KillSwitchIn(BaseModel):
    engaged: bool


def pipeline():
    with _lock:
        if "pipeline" not in _state:
            from .pipeline import Pipeline
            _state["pipeline"] = Pipeline()
        return _state["pipeline"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline()  # load models at start-up, not on the first customer's request
    yield


app = FastAPI(title="CloudServe support triage", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def limit_body(request: Request, call_next):
    if int(request.headers.get("content-length") or 0) > MAX_REQUEST_BYTES:
        return JSONResponse({"detail": "request too large"}, status_code=413)
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse({"detail": "internal error; the ticket was not lost — retry or contact support"}, status_code=500)


@app.post("/tickets")
def process_ticket(t: TicketIn):
    o = pipeline().process(normalise(t.model_dump()), run_id="api")
    return {
        "ticket_id": o.ticket_id, "action": o.action.value, "reason": o.reason,
        "intent": o.classification.intent.model_dump() if o.classification else None,
        "urgency": o.classification.urgency.model_dump() if o.classification else None,
        "response": o.response_text,
        "escalation_summary": o.escalation_summary or None,
        "sources": [{"passage_id": p.passage_id, "title": p.title, "section": p.section, "score": p.score}
                    for p in o.passages],
        "guardrails": [g.model_dump() for g in o.guardrails],
        "latency_ms": o.latency_ms,
    }


@app.get("/decisions/{ticket_id}")
def decisions(ticket_id: str):
    rows = pipeline().dlog.for_ticket(ticket_id[:64])
    if not rows:
        raise HTTPException(404, "no decisions recorded for that ticket")
    return rows


@app.get("/health")
def health():
    p = pipeline()
    on = not p.router.kill_switch_on()
    KILL_SWITCH.set(1 if on else 0)
    return {"status": "ok", "auto_respond_enabled": on, "classifier": p.classifier.version,
            "llm_provider": p.llm.provider, "llm_circuit": getattr(p.llm, "breaker", None) and p.llm.breaker.state}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _require_admin(authorization: str = Header(default="")):
    token = os.getenv("ADMIN_TOKEN", "")
    if not token:
        raise HTTPException(403, "admin endpoints are disabled (ADMIN_TOKEN not set)")
    supplied = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied.encode(), token.encode()):
        raise HTTPException(401, "invalid admin token")


@app.post("/admin/kill-switch", dependencies=[Depends(_require_admin)])
def kill_switch(k: KillSwitchIn):
    from .killswitch import set_kill_switch
    set_kill_switch(k.engaged, actor="api")
    return {"auto_respond_enabled": not k.engaged}


def main():
    logging.basicConfig(level=settings.log_level)
    uvicorn.run(app, host=os.getenv("API_HOST", "127.0.0.1"), port=int(os.getenv("API_PORT", "8000")))


if __name__ == "__main__":
    main()
