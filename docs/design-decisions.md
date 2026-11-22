# Design decisions

Each decision records the context, the options considered, what was chosen, the evidence, and what would make us revisit it. Figures are from the development set (500 tickets) unless stated. Re-run the scripts referenced to reproduce them.

## DD-01 Embeddings through ONNX Runtime, not sentence-transformers

* **Context.** The brief specifies all-MiniLM-L6-v2. The usual route (sentence-transformers) pulls in PyTorch: a large, slow, platform-sensitive install — the commonest reason a submission fails the "runs on a machine that is not yours" gate.
* **Chosen.** The ONNX export of the same model, run with onnxruntime + tokenizers. Same pooling and normalisation as sentence-transformers.
* **Evidence.** Install has no PyTorch; model loads in ~2 s; 500 short texts embed in 0.3 s on 2 CPU cores. The archive is SHA-256 pinned.
* **Revisit if** a larger or domain-tuned embedding model is needed.

## DD-02 Chunk by article section; expand to the article at answer time

* **Context.** 29 articles of 860–1,340 characters, all with the same four sections (Symptoms, Common causes, Resolution, Notes). The Dataset Guide warns that splitting inside a resolution sequence gives passages that "retrieve well but read as incomplete". Ines: customers describe problems in words that do not appear in article titles.
* **Options.** Fixed-size chunks; whole articles; sections.
* **Chosen.** One passage per section, embedded with the article title prepended. Search matches sections (the Symptoms section is written in the customer's language); the answer is drafted from the whole matched article(s); citations point at a section id such as `DOC-DEPLOY-001#resolution`.
* **Evidence.** Expected article ranked first for 89.9% of answerable dev tickets, in the top two for 93.8%. "my deployment keeps dying" retrieves the health-check article with no shared keywords. Search p95 ≈ 14 ms.
* **Revisit if** articles become long or unstructured.

## DD-03 Hybrid scoring with an absolute relevance floor

* **Chosen.** Cosine similarity is the relevance gate (comparable across queries); BM25 adds a bounded boost for exact terms (error codes, "401", CLI commands). Below `RETRIEVAL_MIN_SCORE` nothing is returned.
* **Finding that shaped routing.** Retrieval score does **not** separate answerable from unanswerable tickets: the median best score is 0.73 for answerable and 0.71 for unanswerable (75th percentile 0.81 — higher). Unanswerable tickets (feature requests, roadmap questions) are on-topic, so they match relevant-looking articles. A similarity threshold alone would auto-answer them. "Is this answerable?" therefore gets its own classifier head (DD-04) and the generator may decline (DD-06).

## DD-04 A local, calibrated classifier instead of asking the LLM

* **Chosen.** Multinomial logistic regression on MiniLM embeddings, three heads: intent (22), urgency (3), answerable (2). Temperature-scaled on out-of-fold predictions. Stored as arrays + JSON (no pickle).
* **Why not the LLM.** A5 requires identical decisions for identical input; free-tier LLMs are rate-limited and occasionally unavailable; LLM self-reported confidence is not calibrated. The classifier is deterministic, ~5 ms, free, and keeps routing working during an outage.
* **Evaluation method matters.** The 500 tickets contain only 216 distinct bodies, and 42 of the 80 validation bodies also appear in development. Cross-validation is therefore grouped by ticket body (GroupKFold) so the same wording never sits on both sides of a split. Grouping by full text (subject + body) still let the same body land in training and test folds and reported 98.4% intent accuracy; grouping by body gives 94.4% — the honest figure.
* **Results (grouped 5-fold CV).** Intent accuracy 94.4%, macro precision 94.9%, macro recall 94.1%; calibration in the 0.8–1.0 band (463 of 500 tickets): stated 0.98, observed 0.97. Urgency accuracy 48% (three classes); answerable 73%.
* **Label noise sets a ceiling.** Intent is consistent for identical wording, but urgency and answerability are not: 41 bodies carry conflicting answerability labels. A perfect text-only system could reach at most 86.6% on answerability and 88.8% on routing. Urgency is therefore used for prioritisation and one incident rule, not as a gate for automatic answers.

## DD-05 Routing as a readable, rule-ordered policy

* **Chosen.** `config/routing_policy.json`, evaluated in order, first match decides, rule id + reason logged:
  R0 kill switch · R1 never-auto intents (security incident, compliance request, feature request, unclear request) · R1b sensitive-topic tripwire (compromise, breach, dispute, legal — independent of the model) · R1c security/compliance as a runner-up ≥ 10% · R3 intent confidence < 0.80 · R2 high-urgency live incident (performance or database) · R4 answerable probability < 0.60 · R5 no relevant passage · otherwise auto-respond.
* **Evidence for R1.** In the development labels these four intents are exactly the `must_not_auto_respond` set (87 tickets); Sofia and Daniel both name security as always-escalate; Ines notes feature requests have no documentation.
* **Evidence for R1b.** Found by a test: "someone else has accessed our account and created keys we do not recognise" was classified as `api_key_issue` (0.86) with `security_incident` at 0.0985 and would have been auto-answered. On the development set the tripwire matches 18 tickets, all genuine security incidents (no false positives).
* **Evidence for R2.** All 15 tickets labelled "answerable but escalate" (outside the never-auto intents) are high-urgency performance or database incidents.
* **Thresholds** — see `thresholds.md`.

## DD-06 Generation constrained to the evidence, and allowed to decline

* JSON output; every factual sentence cites a passage id; ids not among the supplied passages are discarded; an answer with no surviving citation is treated as "insufficient evidence" and escalated.
* Customer text sits inside `<ticket>` with angle brackets neutralised; rules live only in the system message; tickets that look like instructions never reach the model (input guardrail).
* Replies disclose that they were drafted automatically and how to reach a person (Ravi: "If I know a machine drafted it I will verify first. Hiding that would be the thing that annoys me.").

## DD-07 Guardrails block; escalations carry a handover

* Six output guardrails run on every draft, none can be disabled by a flag, and an error inside a guardrail fails closed. Private data is never redacted-and-sent: the draft is blocked.
* Every escalation carries: what the ticket seems to be about and the alternatives considered, why it escalated, the checks that stopped it, the documentation found, and any unsent draft (Daniel: "I do not need it to be right. I need it to show its working.").

## DD-08 Decision log as an audit record

* SQLite, one table, append-only (triggers reject UPDATE/DELETE), each record hash-chained to the previous one; `--verify` re-walks the chain. Input summaries are PII-masked. One `final` record per ticket makes reconciliation a single count (A8).

## DD-09 Resilience for free tiers

* Timeouts, bounded retries with jitter, `Retry-After`, circuit breaker (4 failures → 30 s fail-fast), pacing between calls, and a response cache. Rate-limit waiting is excluded from latency and reported separately.
