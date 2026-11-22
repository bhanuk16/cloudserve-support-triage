"""Typed contracts passed between components.

Each stage reads and writes these objects rather than loose dicts, so a change in
one component that breaks the contract fails loudly at the boundary.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Channel(str, Enum):
    EMAIL = "email"
    CHAT = "chat"
    DOCS_COMMENT = "docs_comment"
    FORUM = "forum"
    UNKNOWN = "unknown"


class Action(str, Enum):
    AUTO_RESPOND = "auto_respond"
    ESCALATE = "escalate"
    BLOCK = "block"  # a guardrail stopped the drafted answer; the ticket is escalated


class Ticket(BaseModel):
    """One ticket, normalised to the same shape whatever channel it arrived on."""

    ticket_id: str
    channel: Channel
    subject: str = ""
    body: str = ""
    text: str = ""  # cleaned subject + body; what downstream components read
    received_at: datetime | None = None
    customer_id: str = ""
    customer_name: str = ""
    customer_tier: str = "unknown"
    customer_region: str = "unknown"
    language_fluency: str = "unknown"
    ingest_warnings: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)  # original payload, untouched


class Scored(BaseModel):
    value: str
    confidence: float


class Classification(BaseModel):
    intent: Scored
    urgency: Scored
    answerable: Scored | None = None  # probability the docs contain the answer
    intent_alternatives: list[Scored] = Field(default_factory=list)
    model_version: str = ""
    fallback: bool = False  # true when the classifier could not run and a default was used


class Passage(BaseModel):
    passage_id: str  # e.g. DOC-AUTH-001#resolution — resolves to a real section of the corpus
    doc_id: str
    title: str
    section: str
    text: str
    score: float


class Citation(BaseModel):
    passage_id: str
    doc_id: str
    claim: str


class DraftAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    insufficient_evidence: bool = False
    prompt_version: str = ""
    model: str = ""
    from_cache: bool = False
    throttle_ms: float = 0.0


class GuardrailResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class Outcome(BaseModel):
    """Everything the system decided about one ticket. One per ticket, always."""

    ticket_id: str
    channel: str
    action: Action
    reason: str
    classification: Classification | None = None
    passages: list[Passage] = Field(default_factory=list)
    draft: DraftAnswer | None = None
    response_text: str = ""  # what the customer receives (auto) or the agent handover (escalation)
    escalation_summary: str = ""
    guardrails: list[GuardrailResult] = Field(default_factory=list)
    threshold_applied: float | None = None
    latency_ms: float = 0.0  # processing time, excluding time queued behind the free-tier rate limiter
    throttle_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
