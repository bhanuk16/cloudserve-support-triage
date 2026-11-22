"""Route: decide, before any text is generated, whether this ticket may be answered
automatically. Deterministic by construction: pure rules over the classifier's
(deterministic) output, evaluated in a fixed order. The first rule that fires
decides, and its id and plain-English reason go into the decision log.

Generation and validation can still downgrade an auto-respond decision to an
escalation later (insufficient evidence, guardrail block) — never the reverse.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import ROOT, settings
from .schemas import Action, Classification, Passage

POLICY_PATH = ROOT / "config" / "routing_policy.json"


@dataclass(frozen=True)
class RouteDecision:
    action: Action
    rule_id: str
    reason: str
    threshold_applied: float | None


class Router:
    def __init__(self, policy_path: Path | None = None):
        self.policy = json.loads(Path(policy_path or POLICY_PATH).read_text(encoding="utf-8"))
        p = self.policy
        self.never_auto = set(p["never_auto_intents"]["intents"])
        self.incident = set(p["incident_intents"]["intents"])
        self.t_intent = float(p["intent_confidence_threshold"]["value"])
        self.t_answerable = float(p["answerable_threshold"]["value"])
        self.version = p.get("version", "unversioned")
        trip = p.get("sensitive_topic_tripwire", {})
        self.tripwire = re.compile("|".join(f"(?:{x})" for x in trip.get("patterns", [])), re.I) if trip else None
        ru = p.get("sensitive_runner_up", {})
        self.runner_up = set(ru.get("intents", []))
        self.runner_up_min = float(ru.get("min_probability", 1.1))

    @staticmethod
    def kill_switch_on() -> bool:
        return (not settings.auto_respond_enabled) or settings.kill_switch_file.exists()

    def decide(self, c: Classification, passages: list[Passage], text: str = "") -> RouteDecision:
        p = self.policy
        esc = lambda rule, extra="", t=None: RouteDecision(  # noqa: E731
            Action.ESCALATE, p[rule]["rule_id"], (p[rule]["reason"] + (" " + extra if extra else "")).strip(), t)

        if self.kill_switch_on():
            return esc("kill_switch")
        if c.fallback:
            return esc("intent_confidence_threshold", "The classifier could not run, so no confidence was available.",
                       self.t_intent)
        if c.intent.value in self.never_auto:
            return esc("never_auto_intents", f"(Detected: {c.intent.value.replace('_', ' ')}.)")
        if self.tripwire and (m := self.tripwire.search(text or "")):
            return esc("sensitive_topic_tripwire", f"(Matched: \"{m.group(0)}\".)")
        flagged = [a for a in c.intent_alternatives if a.value in self.runner_up and a.confidence >= self.runner_up_min]
        if flagged:
            return esc("sensitive_runner_up", f"({flagged[0].value.replace('_', ' ')} at {flagged[0].confidence:.0%}.)")
        if c.intent.confidence < self.t_intent:
            alts = ", ".join(f"{a.value} {a.confidence:.0%}" for a in c.intent_alternatives[:2])
            return esc("intent_confidence_threshold",
                       f"Best guess {c.intent.value} at {c.intent.confidence:.0%}, below the {self.t_intent:.0%} "
                       f"threshold. Other candidates: {alts}.", self.t_intent)
        inc = p["incident_intents"]
        if (c.intent.value in self.incident and c.urgency.value == inc["urgency"]
                and c.urgency.confidence >= inc["min_urgency_confidence"]):
            return esc("incident_intents", f"(Detected: {c.intent.value.replace('_', ' ')}, urgency high "
                                           f"at {c.urgency.confidence:.0%}.)")
        p_ans = c.answerable.confidence if c.answerable else 0.0
        if p_ans < self.t_answerable:
            return esc("answerable_threshold", f"Estimated {p_ans:.0%} chance the documentation covers it; "
                                               f"threshold is {self.t_answerable:.0%}.", self.t_answerable)
        if not passages:
            return esc("retrieval")
        return RouteDecision(Action.AUTO_RESPOND, "AUTO",
                             f"Classified as {c.intent.value.replace('_', ' ')} at {c.intent.confidence:.0%} "
                             f"confidence, {p_ans:.0%} likely answerable, and {len(passages)} relevant documentation "
                             f"passage(s) found (best: {passages[0].passage_id}).", self.t_intent)
