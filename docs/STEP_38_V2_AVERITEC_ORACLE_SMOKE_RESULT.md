# Step 38 — AVeriTeC oracle-evidence smoke result

## Run integrity

- Gold: 25
- Predictions: 25
- Paired: 25
- Missing/extra: 0/0
- Stable unique IDs: confirmed before inference
- Evidence condition: annotated question-answer evidence (oracle)
- Model: cached Qwen3-4B-Instruct-2507

## Result

- Accuracy: **0.6400** (16/25)
- Macro-F1: **0.49485**
- Average latency: **807.0 ms/example**
- Average generated tokens: **32.0**

Correct predictions by gold label:

- SUPPORTED: 3/3
- REFUTED: 11/17
- NOT ENOUGH INFO: 2/2
- CONFLICTING: 0/3

The model emitted no `CONFLICTING` prediction. REFUTED produced the largest
absolute error mass: five cases changed to NEI and one to SUPPORTED.

## Interpretation

This 25-example run validates four-label data flow, model loading, pairing, and
evaluation. It is too small and label-imbalanced for a research conclusion.
Because it uses gold QA evidence, it is not an end-to-end AVeriTeC score.

All 25 records reached the configured maximum of 32 tokens. Inspection confirmed
that the label occurs first and the following explanation is truncated. Parsing
is therefore intact, but generation cost is wasted and rationales are incomplete.
First-token maximum probability is also
not calibrated class confidence and must not drive V2 routing as-is.

## Decision

Do not run all 500 development examples yet. Compare short label generation and
explicit label-likelihood scoring on the same frozen 25 examples, with an
AVeriTeC-specific definition of CONFLICTING. Select the verifier protocol before
any full baseline or error-memory experiment.
