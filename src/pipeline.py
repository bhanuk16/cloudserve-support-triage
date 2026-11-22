"""The end-to-end pipeline as a LangGraph state graph.

    input_guard ─┬─(blocked)──────────────────────────────────────────┐
                 └─ classify ─ retrieve ─ route ─┬─(escalate)─────────┤
                                                 └─ generate ─ validate ┴─ finalise

Guarantees:
  * every ticket ends in exactly one stage='final' decision record — escalate,
    block or auto_respond — whatever fails along the way (A8, A9);
  * any exception inside a node is caught, recorded, and turns into an escalation
    with context; the run never stops (A11);
  * decisions only ever move toward a human: generation/validation can downgrade
    an auto_respond, nothing can upgrade an escalation.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from . import guardrails
from .classify import Classifier, fallback, get_classifier
from .decision_log import DecisionLog
from .generate import LLMUnavailable, generate
from .llm import LLMClient, get_llm
from .monitoring import record_outcome
from .retrieval import Retriever, get_retriever
from .route import Router
from .schemas import Action, Classification, DraftAnswer, GuardrailResult, Outcome, Passage, Ticket

log = logging.getLogger(__name__)

# Provisional requirement ids — align these with your PRD (docs/traceability.md).
REQ = {
    "ingest": "FR-01", "classify": "FR-02", "retrieve": "FR-03", "route": "FR-04", "generate": "FR-05",
    "cite": "FR-06", "handover": "FR-07", "validate": "FR-08", "log": "NFR-05", "privacy": "NFR-04",
    "disclose": "FR-09",
}

DISCLOSURE = ("\n\n—\nThis reply was drafted automatically from CloudServe's support documentation. "
              "If it does not resolve your issue, reply to this message and a member of the support team will pick it up.")
ESCALATION_ACK = ("Thanks for getting in touch. Your request has been passed to a member of the CloudServe support team, "
                  "together with the details you sent, so you will not need to repeat them.")


class State(TypedDict, total=False):
    run_id: str
    ticket: Ticket
    started: float
    classification: Classification
    passages: list[Passage]
    context: list[Passage]
    action: Action
    rule_id: str
    reason: str
    threshold: float | None
    draft: DraftAnswer
    guardrails: list[GuardrailResult]
    errors: list[str]
    outcome: Outcome


class Pipeline:
    def __init__(self, classifier: Classifier | None = None, retriever: Retriever | None = None,
                 llm: LLMClient | None = None, decision_log: DecisionLog | None = None, router: Router | None = None):
        self.classifier = classifier or get_classifier()
        self.retriever = retriever or get_retriever()
        self.llm = llm or get_llm()
        self.dlog = decision_log or DecisionLog()
        self.router = router or Router()
        self.graph = self._build()

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(State)
        g.add_node("input_guard", self._safe(self.n_input_guard))
        g.add_node("classify", self._safe(self.n_classify))
        g.add_node("retrieve", self._safe(self.n_retrieve))
        g.add_node("route", self._safe(self.n_route))
        g.add_node("generate", self._safe(self.n_generate))
        g.add_node("validate", self._safe(self.n_validate))
        g.add_node("finalise", self.n_finalise)
        g.set_entry_point("input_guard")
        g.add_conditional_edges("input_guard", lambda s: "finalise" if s.get("action") else "classify")
        g.add_edge("classify", "retrieve")
        g.add_edge("retrieve", "route")
        g.add_conditional_edges("route", lambda s: "generate" if s.get("action") == Action.AUTO_RESPOND else "finalise")
        g.add_conditional_edges("generate", lambda s: "validate" if s.get("action") == Action.AUTO_RESPOND else "finalise")
        g.add_edge("validate", "finalise")
        g.add_edge("finalise", END)
        return g.compile()

    def _safe(self, fn):
        """Wrap a node: an exception escalates with context instead of stopping the run."""
        name = fn.__name__.removeprefix("n_")

        def wrapper(state: State) -> dict:
            try:
                return fn(state)
            except Exception as exc:  # noqa: BLE001
                log.exception("node %s failed on %s", name, state["ticket"].ticket_id)
                return {"action": Action.ESCALATE, "rule_id": "ERR",
                        "reason": f"The {name} step failed ({type(exc).__name__}); handed to a person with what "
                                  f"had been gathered so far.",
                        "errors": state.get("errors", []) + [f"{name}: {type(exc).__name__}: {str(exc)[:200]}"]}
        wrapper.__name__ = fn.__name__
        return wrapper

    # ------------------------------------------------------------------ nodes
    def n_input_guard(self, s: State) -> dict:
        t = s["ticket"]
        if not t.text.strip():
            return {"action": Action.ESCALATE, "rule_id": "IN-EMPTY",
                    "reason": "The ticket arrived with no readable text, so there is nothing to answer automatically."}
        res = guardrails.check_input_integrity(t)
        self.dlog.write(run_id=s["run_id"], ticket_id=t.ticket_id, stage="validation", input_summary=t.text,
                        action_taken=None if res.passed else Action.BLOCK.value,
                        reason="input integrity check " + ("passed" if res.passed else "failed: " + res.detail),
                        guardrail_results={res.name: "pass" if res.passed else f"fail: {res.detail}"},
                        requirement_ids=[REQ["validate"]])
        if not res.passed:
            return {"action": Action.BLOCK, "rule_id": "G-INPUT", "guardrails": [res],
                    "reason": "The ticket contains text that tries to instruct the system, so it was not sent to the "
                              "model. Escalated to a person and recorded for review."}
        return {"guardrails": [res]}

    def n_classify(self, s: State) -> dict:
        t = s["ticket"]
        c = self.classifier.predict(t)
        self.dlog.write(run_id=s["run_id"], ticket_id=t.ticket_id, stage="classification", input_summary=t.text,
                        model={"name": "logreg-minilm", "version": c.model_version},
                        prediction={"intent": c.intent.model_dump(), "urgency": c.urgency.model_dump(),
                                    "answerable": c.answerable.model_dump() if c.answerable else None},
                        alternatives=[a.model_dump() for a in c.intent_alternatives],
                        reason="fallback classification used" if c.fallback else "classified",
                        requirement_ids=[REQ["classify"]])
        return {"classification": c}

    def n_retrieve(self, s: State) -> dict:
        t = s["ticket"]
        passages = self.retriever.search(t.text)
        context = self.retriever.expand_to_documents(passages) if passages else []
        return {"passages": passages, "context": context}

    def n_route(self, s: State) -> dict:
        t, c = s["ticket"], s.get("classification") or fallback("missing classification")
        d = self.router.decide(c, s.get("passages", []), t.text)
        self.dlog.write(run_id=s["run_id"], ticket_id=t.ticket_id, stage="routing",
                        model={"name": "routing-policy", "version": self.router.version},
                        prediction={"intent": c.intent.value, "confidence": c.intent.confidence},
                        sources_used=[{"doc_id": p.passage_id, "score": p.score} for p in s.get("passages", [])],
                        threshold_applied=d.threshold_applied, action_taken=d.action.value,
                        reason=f"[{d.rule_id}] {d.reason}", requirement_ids=[REQ["route"]])
        return {"action": d.action, "rule_id": d.rule_id, "reason": d.reason, "threshold": d.threshold_applied}

    def n_generate(self, s: State) -> dict:
        t = s["ticket"]
        try:
            draft = generate(t, s.get("context") or s.get("passages", []), llm=self.llm)
        except LLMUnavailable as exc:
            self.dlog.write(run_id=s["run_id"], ticket_id=t.ticket_id, stage="generation",
                            model={"name": self.llm.model, "provider": self.llm.provider},
                            action_taken=Action.ESCALATE.value, reason=f"model provider unavailable: {exc}",
                            requirement_ids=[REQ["generate"]])
            return {"action": Action.ESCALATE, "rule_id": "LLM-DOWN",
                    "reason": "The answer could not be drafted because the language model service was unavailable. "
                              "Escalated with the classification and documentation already attached.",
                    "errors": s.get("errors", []) + [f"llm: {exc}"]}
        upd: dict[str, Any] = {"draft": draft}
        reason = "draft produced"
        if draft.insufficient_evidence:
            upd.update(action=Action.ESCALATE, rule_id="GEN-NOEVIDENCE",
                       reason="The documentation retrieved did not contain enough to answer this safely, so the system "
                              "declined to answer rather than guess.")
            reason = "model reported insufficient evidence"
        self.dlog.write(run_id=s["run_id"], ticket_id=t.ticket_id, stage="generation",
                        model={"name": draft.model, "provider": self.llm.provider, "cached": draft.from_cache},
                        prediction={"insufficient_evidence": draft.insufficient_evidence,
                                    "answer_chars": len(draft.answer)},
                        sources_used=[{"doc_id": c.passage_id} for c in draft.citations],
                        action_taken=(upd.get("action") or Action.AUTO_RESPOND).value, reason=reason,
                        prompt_version=draft.prompt_version, requirement_ids=[REQ["generate"], REQ["cite"]])
        return upd

    def n_validate(self, s: State) -> dict:
        t, c, draft = s["ticket"], s["classification"], s["draft"]
        v = guardrails.validate(draft, s.get("context") or s.get("passages", []), t,
                                confidence=c.intent.confidence, threshold=s.get("threshold"))
        results = {r.name: "pass" if r.passed else f"fail: {r.detail}" for r in v.results}
        self.dlog.write(run_id=s["run_id"], ticket_id=t.ticket_id, stage="validation",
                        action_taken=Action.AUTO_RESPOND.value if v.passed else Action.BLOCK.value,
                        reason="all output guardrails passed" if v.passed else
                        "blocked by: " + ", ".join(r.name for r in v.failed),
                        guardrail_results=results, prompt_version=draft.prompt_version,
                        requirement_ids=[REQ["validate"], REQ["privacy"]])
        upd: dict[str, Any] = {"guardrails": s.get("guardrails", []) + v.results}
        if not v.passed:
            names = ", ".join(r.name.replace("_", " ") for r in v.failed)
            upd.update(action=Action.BLOCK, rule_id="G-OUTPUT",
                       reason=f"A drafted answer was stopped before sending by the {names} check. Escalated to a person "
                              f"with the draft and the reason attached.")
        return upd

    def n_finalise(self, s: State) -> dict:
        t = s["ticket"]
        action = s.get("action") or Action.ESCALATE
        c = s.get("classification")
        passages = s.get("passages", [])
        draft = s.get("draft")
        throttle = draft.throttle_ms if draft else 0.0
        latency = (time.perf_counter() - s["started"]) * 1000 - throttle
        if action == Action.AUTO_RESPOND and draft:
            response = render_answer(draft.answer, s.get("context") or passages) + DISCLOSURE
            summary = ""
        else:
            response = ESCALATION_ACK
            summary = escalation_summary(t, c, passages, draft, s.get("reason", ""), s.get("guardrails", []))
        outcome = Outcome(
            ticket_id=t.ticket_id, channel=t.channel.value, action=action, reason=s.get("reason", ""),
            classification=c, passages=passages, draft=draft, response_text=response, escalation_summary=summary,
            guardrails=s.get("guardrails", []), threshold_applied=s.get("threshold"), latency_ms=round(latency, 1), throttle_ms=round(throttle, 1),
            errors=s.get("errors", []) + [f"ingest: {w}" for w in t.ingest_warnings],
            requirement_ids=[REQ["log"]],
        )
        self.dlog.write(
            run_id=s["run_id"], ticket_id=t.ticket_id, stage="final", input_summary=t.text,
            model={"classifier": c.model_version if c else None, "llm": draft.model if draft else None},
            prediction={"intent": c.intent.value, "confidence": c.intent.confidence,
                        "urgency": c.urgency.value} if c else None,
            alternatives=[a.model_dump() for a in c.intent_alternatives] if c else None,
            sources_used=[{"doc_id": p.passage_id, "score": p.score} for p in passages],
            threshold_applied=s.get("threshold"), action_taken=action.value,
            reason=f"[{s.get('rule_id', 'ERR')}] {s.get('reason', '')}",
            guardrail_results={g.name: "pass" if g.passed else f"fail: {g.detail}" for g in s.get("guardrails", [])},
            prompt_version=draft.prompt_version if draft else None,
            requirement_ids=sorted(set(outcome.requirement_ids + [REQ["route"]])),
        )
        record_outcome(outcome, c)
        return {"outcome": outcome}

    # ------------------------------------------------------------------ entry
    def process(self, ticket: Ticket, run_id: str | None = None) -> Outcome:
        state: State = {"run_id": run_id or f"adhoc-{uuid.uuid4().hex[:8]}", "ticket": ticket,
                        "started": time.perf_counter(), "errors": []}
        try:
            return self.graph.invoke(state)["outcome"]
        except Exception as exc:  # last line of defence: still produce and log a decision
            log.exception("pipeline crashed on %s", ticket.ticket_id)
            state.update(action=Action.ESCALATE, rule_id="ERR",
                         reason=f"Unexpected failure ({type(exc).__name__}); escalated.",
                         errors=[f"pipeline: {type(exc).__name__}: {str(exc)[:200]}"])
            return self.n_finalise(state)["outcome"]


# ---------------------------------------------------------------------- rendering
def render_answer(answer: str, passages: list[Passage]) -> str:
    """Turn inline [passage-id] markers into numbered footnotes a customer can follow."""
    import re

    by_id = {p.passage_id: p for p in passages}
    order: list[str] = []

    def sub(m):
        pid = m.group(1)
        if pid not in by_id:
            return ""
        if pid not in order:
            order.append(pid)
        return f" [{order.index(pid) + 1}]"

    body = re.sub(r"\s*\[([A-Z]+-[A-Z]+-\d{3}#[a-z_]+)\]", sub, answer).strip()
    refs = "\n".join(f"[{i + 1}] {by_id[p].title} — {by_id[p].section} ({by_id[p].doc_id})" for i, p in enumerate(order))
    return f"{body}\n\nSources:\n{refs}" if refs else body


def escalation_summary(t: Ticket, c: Classification | None, passages: list[Passage], draft: DraftAnswer | None,
                       reason: str, checks: list[GuardrailResult]) -> str:
    """The handover a tier-two engineer asked for: what it is, what was found, what we were unsure about."""
    lines = [f"Ticket {t.ticket_id} via {t.channel.value} — {t.customer_tier} customer, {t.customer_region}, "
             f"language: {t.language_fluency}."]
    if c:
        alts = ", ".join(f"{a.value} ({a.confidence:.0%})" for a in c.intent_alternatives[:2])
        lines.append(f"What it seems to be about: {c.intent.value.replace('_', ' ')} ({c.intent.confidence:.0%} confident"
                     f"{'; also considered ' + alts if alts else ''}). Urgency: {c.urgency.value}"
                     f" ({c.urgency.confidence:.0%}).")
    lines.append(f"Why it came to you: {reason}")
    failed = [g for g in checks if not g.passed]
    if failed:
        lines.append("Checks that stopped it: " + "; ".join(f"{g.name.replace('_', ' ')} — {g.detail}" for g in failed))
    if passages:
        lines.append("Documentation that looked relevant: " + "; ".join(
            f"{p.title} › {p.section} ({p.passage_id}, match {p.score:.2f})" for p in passages[:3]))
    else:
        lines.append("Documentation: nothing relevant enough was found in the knowledge base.")
    if draft and draft.answer:
        lines.append("Unsent draft for your review (not sent to the customer):\n" + draft.answer)
    return "\n".join(lines)
