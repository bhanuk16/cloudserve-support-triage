"""Generate: draft a grounded, cited answer from the retrieved passages.

Instruction/data separation (Build Spec, Generate): our rules live only in the
system message; the ticket goes in the user message inside <ticket> tags with any
angle brackets neutralised, so a customer cannot close the tag and smuggle in new
"instructions". The model's output is parsed as JSON and every citation is checked
against the passages we actually supplied: invented ids are discarded (A6).
"""
from __future__ import annotations

import json
import re

from .llm import LLMClient, LLMUnavailable, get_llm
from .prompts import load_prompt
from .schemas import Citation, DraftAnswer, Passage, Ticket

PROMPT_FILE = "generate_answer.v1.md"
_MARKER = re.compile(r"\[([A-Z]+-[A-Z]+-\d{3}#[a-z_]+)\]")


def _neutralise(text: str) -> str:
    return text.replace("<", "‹").replace(">", "›")


def build_messages(ticket: Ticket, passages: list[Passage]) -> tuple[list[dict], str]:
    prompt = load_prompt(PROMPT_FILE)
    docs = "\n".join(
        f'<passage id="{p.passage_id}">\n{p.title} — {p.section}\n{_neutralise(p.text)}\n</passage>' for p in passages
    )
    user = prompt.user_template.format(channel=ticket.channel.value, ticket_text=_neutralise(ticket.text), passages=docs)
    return [{"role": "system", "content": prompt.system}, {"role": "user", "content": user}], prompt.tag


def parse_output(raw: str) -> dict:
    """Tolerant JSON extraction: models sometimes wrap JSON in prose or code fences."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError("model output was not valid JSON")


def generate(ticket: Ticket, passages: list[Passage], llm: LLMClient | None = None) -> DraftAnswer:
    """Raises LLMUnavailable if the provider cannot answer; the pipeline escalates on that."""
    llm = llm or get_llm()
    messages, prompt_tag = build_messages(ticket, passages)
    resp = llm.chat(messages)
    try:
        data = parse_output(resp.content)
    except ValueError:
        return DraftAnswer(answer="", insufficient_evidence=True, prompt_version=prompt_tag, model=resp.model,
                           from_cache=resp.from_cache, throttle_ms=resp.throttle_ms)

    supplied = {p.passage_id: p for p in passages}
    answer = str(data.get("answer") or "").strip()
    citations: list[Citation] = []
    for c in data.get("citations") or []:
        pid = str((c or {}).get("passage_id", "")).strip()
        if pid in supplied:  # only ids we actually retrieved survive
            citations.append(Citation(passage_id=pid, doc_id=supplied[pid].doc_id, claim=str(c.get("claim", ""))[:500]))
    # inline markers count as citations too, if they resolve
    for pid in _MARKER.findall(answer):
        if pid in supplied and pid not in {c.passage_id for c in citations}:
            citations.append(Citation(passage_id=pid, doc_id=supplied[pid].doc_id, claim=""))

    insufficient = bool(data.get("insufficient_evidence")) or not answer or not citations
    return DraftAnswer(answer=answer, citations=citations, insufficient_evidence=insufficient,
                       prompt_version=prompt_tag, model=resp.model, from_cache=resp.from_cache,
                       throttle_ms=resp.throttle_ms)


__all__ = ["generate", "build_messages", "parse_output", "LLMUnavailable"]
