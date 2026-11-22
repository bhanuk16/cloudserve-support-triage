"""Validate: checks that run on every drafted response before release. Any failure
blocks the response and escalates the ticket with the reason attached. A guardrail
that only warns is not a guardrail (Build Spec, Validate).

  input_integrity   — the ticket tries to give the system instructions (checked
                      before generation; such tickets are never sent to the model)
  private_data      — emails, phone numbers, keys/tokens, card numbers, customer ids,
                      the customer's own name: block and escalate, never redact-and-send
  citation_validity — every citation resolves to a passage we actually retrieved (A6)
  grounding         — every factual sentence is cited and semantically supported by the
                      passage it cites; numbers must appear in the cited text
  scope_commitments — no refunds, credits, deadlines, "fixed on our side", or claims
                      that the system acted on the account
  prompt_leakage    — the reply does not echo our instructions
  confidence_floor  — a routing threshold was actually applied to a real confidence

Every result (pass or fail, with detail) is written to the decision log.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .embeddings import get_embedder
from .schemas import DraftAnswer, GuardrailResult, Passage, Ticket

import os
# sentence-vs-cited-passage cosine. On the 200 senior-agent references: 0.40 passes 75% of genuinely
# supported (human-paraphrased) sentences and only 8% of sentences compared with an unrelated article.
GROUNDING_MIN_SIM = float(os.getenv("GROUNDING_MIN_SIM", "0.40"))

# ---------------------------------------------------------------------------- patterns
_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (all |any |the )?(previous|prior|above|your) (instructions|rules)",
    r"forget (all |your |the )?(previous |prior )?(instructions|rules)",
    r"(reveal|print|show|repeat|output) (me )?(your|the) (system )?(prompt|instructions|rules)",
    r"\byou are now\b", r"\bact as (an? )?(admin|administrator|developer|system|different)",
    r"\bnew (instructions|rules)\s*:", r"\bsystem prompt\b", r"\bjailbreak\b", r"\bDAN mode\b",
    r"</?(system|ticket|passage|documentation|instructions?)>",
    r"\bdeveloper mode\b", r"\boverride (the |your )?(rules|policy|instructions)",
    r"(issue|grant|approve|process) (me |us )?(a |an )?(full )?(refund|credit)s? (immediately|now|without)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION), re.I)

_PII = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?<![\w-])\+?\d[\d\s().-]{8,}\d(?![\w-])"),
    "api_key_or_token": re.compile(r"\b(sk|pk|rk|ghp|gho|xox[abp]|AKIA|AIza)[-_A-Za-z0-9]{12,}|\b[A-Fa-f0-9]{32,}\b|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    "customer_id": re.compile(r"\bCUST-\d{3,}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
}
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")

_COMMITMENT = [
    r"\b(will|has been|have been|we|I)( be)? refund(ed)?\b", r"\brefund (has|will|is being)", r"\bcredit(ed|s)? (has|have|will|to your)", r"\bcompensat",
    r"\b(has|have) been (fixed|resolved|issued|reset|restored|unlocked|applied)\b",
    r"\bwe (have|'ve) (fixed|resolved|reset|restored|unlocked|issued|updated|changed)\b",
    r"\bI (have|'ve) (fixed|resolved|reset|restored|unlocked|issued|updated|changed)\b",
    r"\bwill be (fixed|resolved|deployed|released) (by|on|within|in)\b",
    r"\bwithin \d+ (hours?|days?|business days?)\b(?!.*automatically)",
    r"\bby (monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|end of (day|week))\b",
    r"\bguarantee", r"\bwe promise\b", r"\bwaive", r"\bfree of charge\b",
]
_COMMITMENT_RE = re.compile("|".join(_COMMITMENT), re.I)

_LEAK_MARKERS = ("insufficient_evidence", "Rules — follow all", "<passage", "<ticket", "You are drafting the first reply")
_MARKER = re.compile(r"\s*\[([A-Z]+-[A-Z]+-\d{3}#[a-z_]+)\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_COURTESY = re.compile(r"^(thanks|thank you|hi|hello|sorry|apologies|we understand|i understand|happy to help|"
                       r"let us know|please (let|reply|get)|if (this|that|the issue) (does not|doesn't|persists)|"
                       r"hope this helps)", re.I)


def _luhn(digits: str) -> bool:
    d = [int(c) for c in digits][::-1]
    total = sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2])
    return total % 10 == 0


# ---------------------------------------------------------------------------- checks
def check_input_integrity(ticket: Ticket) -> GuardrailResult:
    m = _INJECTION_RE.search(ticket.text or "")
    if m:
        return GuardrailResult(name="input_integrity", passed=False,
                               detail=f"ticket contains instruction-like text: {m.group(0)[:60]!r}")
    return GuardrailResult(name="input_integrity", passed=True)


def find_private_data(text: str, ticket: Ticket | None = None) -> list[str]:
    found = [k for k, rx in _PII.items() if rx.search(text)]
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn(digits):
            found.append("card_number")
            break
    if ticket and ticket.customer_name:
        for part in ticket.customer_name.split():
            if len(part) >= 3 and re.search(rf"\b{re.escape(part)}\b", text, re.I):
                found.append("customer_name")
                break
    return sorted(set(found))


def check_private_data(answer: str, ticket: Ticket) -> GuardrailResult:
    hits = find_private_data(answer, ticket)
    return GuardrailResult(name="private_data", passed=not hits, detail=", ".join(hits))


def check_citations(draft: DraftAnswer, passages: list[Passage]) -> GuardrailResult:
    supplied = {p.passage_id for p in passages}
    cited = {c.passage_id for c in draft.citations} | set(_MARKER.findall(draft.answer))
    bad = sorted(cited - supplied)
    if not cited:
        return GuardrailResult(name="citation_validity", passed=False, detail="answer carries no citations")
    if bad:
        return GuardrailResult(name="citation_validity", passed=False, detail=f"unresolvable citations: {bad}")
    return GuardrailResult(name="citation_validity", passed=True, detail=f"{len(cited)} citation(s) resolved")


def split_claims(answer: str) -> list[tuple[str, list[str]]]:
    """Split into sentences and attach the passage ids each sentence cites."""
    out = []
    for sent in _SENT_SPLIT.split(answer.strip()):
        ids = _MARKER.findall(sent)
        clean = _MARKER.sub("", sent).strip()
        if clean:
            out.append((clean, ids))
    return out


def check_grounding(draft: DraftAnswer, passages: list[Passage], min_sim: float = GROUNDING_MIN_SIM) -> GuardrailResult:
    by_id = {p.passage_id: p for p in passages}
    claims = split_claims(draft.answer)
    last_ids: list[str] = []
    unsupported: list[str] = []
    to_check: list[tuple[str, list[str]]] = []
    for sent, ids in claims:
        if _COURTESY.match(sent) and not re.search(r"\d", sent):
            continue  # greetings and sign-offs carry no factual claim
        ids = ids or last_ids  # a sentence continuing a cited step inherits its citation
        if not ids:
            unsupported.append(f"uncited: {sent[:80]}")
            continue
        last_ids = ids
        to_check.append((sent, ids))
        # numbers in the claim must literally appear in the cited text
        nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", sent))
        cited_text = " ".join(by_id[i].text for i in ids if i in by_id)
        if nums - set(re.findall(r"\b\d+(?:\.\d+)?\b", cited_text)):
            unsupported.append(f"number not in source: {sent[:80]}")
    if to_check:
        emb = get_embedder()
        sents = emb.embed([s for s, _ in to_check])
        for (sent, ids), v in zip(to_check, sents):
            texts = [f"{by_id[i].title}. {by_id[i].text}" for i in ids if i in by_id]
            if not texts:
                unsupported.append(f"cites unknown passage: {sent[:80]}")
                continue
            # compare with each line of the cited passage: steps are where the support lives
            lines = [ln for t in texts for ln in re.split(r"\n+", t) if len(ln.strip()) > 20] or texts
            best = float(np.max(emb.embed(lines) @ v))
            if best < min_sim:
                unsupported.append(f"weak support ({best:.2f}): {sent[:80]}")
    if not claims:
        return GuardrailResult(name="grounding", passed=False, detail="empty answer")
    return GuardrailResult(name="grounding", passed=not unsupported, detail="; ".join(unsupported[:3]))


def check_scope(answer: str) -> GuardrailResult:
    m = _COMMITMENT_RE.search(answer)
    return GuardrailResult(name="scope_commitments", passed=m is None,
                           detail=f"commitment or action claim: {m.group(0)!r}" if m else "")


def check_prompt_leakage(answer: str) -> GuardrailResult:
    hit = next((m for m in _LEAK_MARKERS if m.lower() in answer.lower()), None)
    return GuardrailResult(name="prompt_leakage", passed=hit is None, detail=f"contains {hit!r}" if hit else "")


def check_confidence_floor(confidence: float | None, threshold: float | None) -> GuardrailResult:
    ok = confidence is not None and threshold is not None and confidence >= threshold
    return GuardrailResult(name="confidence_floor", passed=ok,
                           detail=f"confidence={confidence} threshold={threshold}")


@dataclass
class Validation:
    results: list[GuardrailResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed(self) -> list[GuardrailResult]:
        return [r for r in self.results if not r.passed]


def validate(draft: DraftAnswer, passages: list[Passage], ticket: Ticket,
             confidence: float | None, threshold: float | None) -> Validation:
    """Run every output guardrail. All run (no short-circuit) so the log shows the full picture."""
    checks = [
        lambda: check_private_data(draft.answer, ticket),
        lambda: check_citations(draft, passages),
        lambda: check_grounding(draft, passages),
        lambda: check_scope(draft.answer),
        lambda: check_prompt_leakage(draft.answer),
        lambda: check_confidence_floor(confidence, threshold),
    ]
    results = []
    for fn in checks:
        try:
            results.append(fn())
        except Exception as exc:  # fail closed: a guardrail that errors blocks
            results.append(GuardrailResult(name=getattr(fn, "__name__", "guardrail"), passed=False,
                                           detail=f"guardrail error, failing closed: {type(exc).__name__}"))
    return Validation(results)
