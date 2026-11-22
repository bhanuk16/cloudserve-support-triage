# Threat model

Scope: the triage service, its data stores and its model provider. Method: list what an attacker or an accident could do at each boundary, and name the control in this codebase that stops it.

## Trust boundaries

1. **Customer → ingest.** Untrusted free text from four channels.
2. **Service → model provider.** Ticket text and documentation leave the machine; responses are untrusted.
3. **Documentation corpus → retrieval.** Trusted source, but its content is passed to the model.
4. **Operator → API / CLI.** Admin actions (kill switch).
5. **Repository → anyone who clones it.** Secrets and artefacts.

## Threats and controls

| # | Threat (OWASP LLM ref) | Control | Where | Residual risk |
|---|---|---|---|---|
| T1 | Direct prompt injection in a ticket (LLM01) | Input-integrity guardrail blocks before the model; rules only in system message; ticket fenced in `<ticket>` with `<`/`>` neutralised; output prompt-leakage check | `guardrails.check_input_integrity`, `generate._neutralise` | Novel phrasings evade the pattern list; output guardrails and grounding still apply |
| T2 | Indirect injection via documentation (LLM01) | Corpus is internal and versioned (index fingerprint); passages neutralised the same way | `retrieval`, `generate` | A malicious article edit; mitigated by docs review cycle |
| T3 | Private data in a reply (LLM02) | Prompt forbids it; private-data guardrail blocks (never redacts-and-sends); decision log stores masked summaries only | `guardrails.check_private_data`, `decision_log.mask` | Free-text names not in the ticket record |
| T4 | Hallucinated or unsupported answer (LLM09) | Retrieval-only context; per-sentence citations; invented citations discarded; grounding guardrail (semantic support + numbers must appear in source); generator may decline | `generate`, `guardrails.check_grounding` | Semantically close but subtly wrong sentences; measured by human review sample |
| T5 | Unauthorised commitment (refund, deadline, "fixed") (LLM06) | Scope guardrail; billing disputes and legal matters escalated by tripwire | `guardrails.check_scope`, `route` R1b | — |
| T6 | Security incident handled by a bot | Never-auto intents, independent keyword tripwire, runner-up rule | `route` R1/R1b/R1c | — |
| T7 | Malformed / oversized input, encoding tricks (LLM10) | Ingest never raises; NFKC, control and zero-width characters stripped; length caps; API body limit and schema validation | `ingest`, `api` | — |
| T8 | Provider outage, throttling, cost runaway (LLM10) | Timeouts, retries, circuit breaker, pacing, cache, local classifier keeps routing | `llm` | — |
| T9 | API key leak | `.env` only, git-ignored; never logged or echoed in errors (tested); gitleaks in CI | `config`, `llm`, CI | Key on a developer's disk |
| T10 | Tampered model or artefact (LLM03) | Embedding archive SHA-256 pinned; safe extraction; classifier stored without pickle (`allow_pickle=False`); dependencies pinned | `setup_models`, `classify` | Compromise of a pinned upstream package |
| T11 | Audit log altered to hide a bad decision | Append-only triggers; hash chain with `--verify` | `decision_log` | An attacker with file access can delete the whole database (detectable as a gap against tickets processed) |
| T12 | Unauthorised kill-switch use over HTTP | Disabled unless `ADMIN_TOKEN` set; constant-time comparison; localhost bind by default; audit log | `api`, `killswitch` | — |

## Accepted risks

* Ticket text is sent to a third-party model provider. Before production: a data-processing agreement, or a self-hosted model.
* The tripwire and injection patterns are lists; they are a net, not a proof. They are backed by the output guardrails and by red-team tests, which should be extended with every incident.
