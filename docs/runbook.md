# Operational runbook

Written for someone who did not build the system, at two in the morning.

## Stop automatic answering now (kill switch)

```bash
python -m src.killswitch on        # from the project folder, virtual environment active
python -m src.killswitch status    # confirm: "kill switch ENGAGED"
```
* **Effect:** from the next ticket, every ticket is escalated to a person with its handover. No restart, no deployment.
* **Who may use it:** anyone on the support on-call rota or the head of support. Every use is appended to `storage/kill_switch_audit.log` with time and user.
* **Tickets in flight:** a ticket already past routing finishes its current step but is still validated by every guardrail before anything is sent.
* **Release:** `python -m src.killswitch off` — only after the review step below.
* **Tested by:** `tests/test_acceptance.py::test_kill_switch_escalates_everything`, on every push.

## Incident procedure

| Step | What to do | Who | Target time |
|---|---|---|---|
| 1. Detect | Alert or report: a wrong/unsafe reply, a guardrail spike in Grafana, error rate up, or a customer complaint. Note the ticket id. | Whoever notices | — |
| 2. Contain | Engage the kill switch. If one reply is the problem, also flag the ticket to an agent to send a correction. | On-call | 5 min |
| 3. Assess | `python -m src.decision_log --ticket <id>` shows every decision with model version, prompt version, sources, thresholds and guardrail results. Decide: wrong article (→ docs owner), retrieval/classification error (→ engineering), guardrail gap (→ engineering), outage (→ wait/fallback). `python -m src.decision_log --verify` confirms the log is intact. | On-call + engineer | 30 min |
| 4. Notify | Head of support. If private data was exposed: the data protection lead, same day. Affected customer via an agent. | Head of support | 1 h (4 h for data protection) |
| 5. Remediate | Fix in the right place: article (Ines), policy/threshold (config), prompt (new prompt version), or code. Add the failing ticket to `tests/` so it cannot regress. | Engineer / docs owner | Before release |
| 6. Review | Run the harness on the validation set; compare with the last good report; release the kill switch; record the incident in the governance register and, if requirements changed, in the PRD revision log. | Engineer + head of support | Within 2 working days |

## Common situations

| Symptom | Likely cause | Action |
|---|---|---|
| Every ticket escalates with "language model service was unavailable" | Provider outage, bad key, or free-tier quota | Check `/health` (`llm_circuit`). The system is behaving correctly — agents receive full handovers. Check the key and the provider status page. |
| Run is slow | Rate limiting | Expected on free tiers; `LLM_MIN_INTERVAL_SECONDS` paces calls. Waiting time is excluded from latency figures. |
| Grounding guardrail blocks spike | Documentation changed, or a model change | Check recent article edits; the index rebuilds automatically when the corpus changes. |
| "Embedding model not found" | Clean machine | `python -m src.setup_models` |
| "classifier.json not found" | Clean machine | `python -m src.classify --train` |
