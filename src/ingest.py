"""Ingest: turn a raw ticket from any of the four channels into one Ticket.

Design rules (Build Spec, Ingest):
  * never raise on a bad record — missing fields, odd characters and empty bodies
    become warnings on the ticket, not exceptions;
  * keep the original payload and the channel, because both matter downstream;
  * clean text defensively (unicode normalisation, control/zero-width characters
    stripped, length capped) so hostile or broken input cannot flood the model.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .config import settings
from .schemas import Channel, Ticket

_CHANNEL_ALIASES = {
    "email": Channel.EMAIL, "e-mail": Channel.EMAIL, "mail": Channel.EMAIL,
    "chat": Channel.CHAT, "live_chat": Channel.CHAT, "livechat": Channel.CHAT,
    "docs_comment": Channel.DOCS_COMMENT, "docs": Channel.DOCS_COMMENT,
    "documentation_comment": Channel.DOCS_COMMENT, "doc_comment": Channel.DOCS_COMMENT,
    "forum": Channel.FORUM, "community": Channel.FORUM, "community_forum": Channel.FORUM,
}

# zero-width and bidi-control characters are a common way to hide instructions in text
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_QUOTED_REPLY = re.compile(r"^\s*>.*$", re.MULTILINE)  # quoted earlier messages in email/forum threads
_WS = re.compile(r"[ \t]+")


def clean_text(value: Any, limit: int | None = None) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    text = unicodedata.normalize("NFKC", value)
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub(" ", text)
    text = _WS.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    limit = limit or settings.max_body_chars
    return text[:limit]


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def normalise(raw: Any, index: int = 0) -> Ticket:
    """Normalise one raw record. Always returns a Ticket."""
    warnings: list[str] = []
    if not isinstance(raw, dict):
        warnings.append(f"record was {type(raw).__name__}, not an object")
        raw = {"body": raw if isinstance(raw, str) else ""}

    ticket_id = clean_text(raw.get("ticket_id"), 64) or f"UNKNOWN-{index:05d}"
    if not raw.get("ticket_id"):
        warnings.append("missing ticket_id")

    channel_raw = clean_text(raw.get("channel"), 40).lower().replace(" ", "_").replace("-", "_")
    channel = _CHANNEL_ALIASES.get(channel_raw, Channel.UNKNOWN)
    if channel is Channel.UNKNOWN:
        warnings.append(f"unrecognised channel {channel_raw!r}")

    subject = clean_text(raw.get("subject"), 300)
    body = clean_text(raw.get("body"))
    if channel in (Channel.EMAIL, Channel.FORUM):
        # drop quoted history so we classify what the customer is asking now
        stripped = clean_text(_QUOTED_REPLY.sub("", body))
        body = stripped or body
    if not body:
        warnings.append("empty body")
    if raw.get("body") and len(str(raw.get("body"))) > settings.max_body_chars:
        warnings.append("body truncated")

    received_at = _parse_time(raw.get("received_at"))
    if raw.get("received_at") and received_at is None:
        warnings.append("unparseable received_at")

    text = (f"{subject}\n{body}" if subject else body).strip()

    return Ticket(
        ticket_id=ticket_id,
        channel=channel,
        subject=subject,
        body=body,
        text=text,
        received_at=received_at,
        customer_id=clean_text(raw.get("customer_id"), 64),
        customer_name=clean_text(raw.get("customer_name"), 120),
        customer_tier=clean_text(raw.get("customer_tier"), 30).lower() or "unknown",
        customer_region=clean_text(raw.get("customer_region"), 30).lower() or "unknown",
        language_fluency=clean_text(raw.get("language_fluency"), 30).lower() or "unknown",
        ingest_warnings=warnings,
        raw=raw,
    )


def load_raw(path: str | Path) -> list[Any]:
    """Load a ticket file. Accepts a JSON list, a {"tickets": [...]} object, or JSON Lines."""
    content = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    stripped = content.lstrip()
    if not stripped:
        return []
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            data = data.get("tickets", [data])
        return list(data)
    except json.JSONDecodeError:
        records = []
        for n, line in enumerate(stripped.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"ticket_id": f"MALFORMED-LINE-{n:05d}", "body": line[:2000]})
        return records


def load_tickets(path: str | Path) -> list[Ticket]:
    return [normalise(r, i) for i, r in enumerate(load_raw(path))]


def iter_tickets(records: Iterable[Any]) -> Iterable[Ticket]:
    for i, r in enumerate(records):
        yield normalise(r, i)
