# Thresholds: how each was chosen

All thresholds live in configuration (`config/routing_policy.json`, `.env`) so they can be changed without code. "OOF" means out-of-fold predictions from grouped 5-fold cross-validation on the development set (`storage/artifacts/oof_predictions.json`, written by `python -m src.classify --train`).

| Threshold | Value | Basis | Trade-off accepted |
|---|---|---|---|
| Intent confidence (R3) | 0.80 | Calibration on OOF predictions: in the 0.8–1.0 band stated confidence 0.98 vs observed accuracy 0.97 (n = 463); below 0.8 the classifier is right only 44–70% of the time. | Tickets the classifier is unsure about go to a person rather than risk answering the wrong question (Sofia's non-fluent customers). |
| Answerable probability (R4) | 0.60 | OOF sweep (below): routing agreement peaks around 0.5–0.6; above 0.7 automation collapses. | Some unanswerable tickets pass this gate; the generator's "insufficient evidence" and the grounding guardrail are the second and third lines of defence. |
| Security/compliance runner-up (R1c) | 0.10 | Conservative safety net for wording the training data did not contain. | A few routine tickets may escalate unnecessarily. |
| Incident urgency confidence (R2) | 0.55 | Urgency is only 48% accurate overall; requiring ≥ 0.55 limits the rule to cases the model is relatively sure about. | Some live incidents rely on R3/R4 or the generator instead. |
| Retrieval floor (`RETRIEVAL_MIN_SCORE`) | 0.35 | 5th percentile of the best score for answerable tickets is 0.54; nonsense queries score below 0.35. The floor removes junk, it does not decide answerability (see DD-03). | — |
| Grounding similarity (`GROUNDING_MIN_SIM`) | 0.40 | On the 200 senior-agent reference answers, 0.40 passes 75% of human-paraphrased supported sentences and only 8% of sentences compared against an unrelated article. LLM drafts stay closer to the source than human paraphrase, so the pass rate on real drafts is expected to be higher — **confirm on the first live run** (grounding block rate in the metrics report). | Stricter values cut hallucination risk but raise the escalation rate. |

## OOF routing sweep (development set, classifier-only, before generation and guardrails)

| Intent ≥ | Answerable ≥ | Auto-answer share | Auto-answered but labelled escalate | Never-auto violations |
|---|---|---|---|---|
| 0.80 | 0.00 | 77.0% | 97 | 0 |
| 0.80 | 0.50 | 73.2% | 90 | 0 |
| **0.80** | **0.60** | **66.2%** | **81** | **0** |
| 0.80 | 0.70 | 48.8% | 57 | 0 |
| 0.80 | 0.80 | 23.2% | 22 | 0 |

"Labelled escalate" includes label noise (identical wording labelled both ways); the ceiling for routing agreement from text alone is 88.8%.

## To do after the first live run

Re-check the grounding block rate, the share of "insufficient evidence" declines, and the routing confusion on the **validation** set, then adjust and record any change in the PRD revision log. Do not tune on the hidden set.
