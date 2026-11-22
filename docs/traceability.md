# Traceability: evidence → requirement → code → test

**Fill this in from your own PRD.** The requirement ids below are placeholders used in `src/pipeline.py` (`REQ`) and written into every decision-log record; rename them to match your PRD and keep this table as the visible chain the assessors ask for.

| Requirement id (placeholder) | What it covers | Discovery evidence (your workbook section/row) | Code | Prompt | Test |
|---|---|---|---|---|---|
| FR-01 | Ingest four channels into one representation | | `src/ingest.py` | — | `test_a2_*` |
| FR-02 | Classify intent and urgency with calibrated confidence | | `src/classify.py` | — | `test_a3_*` |
| FR-03 | Retrieve relevant passages; return nothing when nothing is relevant | | `src/retrieval.py` | — | `test_a4_*` |
| FR-04 | Route by policy and data-chosen thresholds; log reason | | `src/route.py`, `config/routing_policy.json` | — | `test_a5_*`, `test_a7_never_auto_*` |
| FR-05 | Grounded answer; decline when evidence is insufficient | | `src/generate.py` | PR-GEN-01 | `test_a6_*` |
| FR-06 | Citations resolve to retrieved passages | | `src/generate.py`, `guardrails.check_citations` | PR-GEN-01 | `test_a6_*` |
| FR-07 | Escalation handover (summary, sources, uncertainty) | | `pipeline.escalation_summary` | — | `test_a9_a10_*` |
| FR-08 | Guardrails run on every reply and can block | | `src/guardrails.py` | — | `test_a7_*` |
| FR-09 | Replies disclose they were drafted automatically | | `pipeline.DISCLOSURE` | — | — |
| NFR-04 | Privacy: no private data in replies or logs | | `guardrails`, `decision_log.mask` | PR-GEN-01 | `test_a7_output_guardrails_block` |
| NFR-05 | Auditability: every decision logged, reconcilable | | `src/decision_log.py` | — | `test_a8_*` |
