# ERRORFlow V3 cross-domain validation

Status: **PROTOCOL FROZEN; SCIFACT EXECUTION NOT RUN**.

## Frozen source system

- Source dataset: AVeriTeC train only.
- B1: immutable `conflict_audit_v1`.
- Action: `conflicting_rescue_v1`.
- Memory: all 3,068 source action outcomes.
- Router: R6, `top_k=5`, `min_support=2`, `cost_weight=0.1`, utility threshold `>0`.
- No SciFact label or outcome may update memory, prompts, similarity weights,
  thresholds, or routing parameters.

## SciFact protocol

The official SciFact release separates claims into train/dev/test JSONL files
and stores evidence abstracts in `corpus.jsonl`. The public test labels are not
available, so the first external evaluation uses official dev labels. Evidence
is restricted to official gold rationale sentences joined from the corpus.
This is an oracle-evidence verifier-transfer test, not an open-retrieval result
and not an official full-pipeline SciFact score.

SciFact has three verdict labels. ERRORFlow retains four-label B1/A3 scoring so
the frozen prompts and AVeriTeC memory remain unchanged. The predeclared primary
projection maps `CONFLICTING` to `NOT ENOUGH INFO`; a strict unprojected result
is retained as sensitivity analysis. Report the number of projected predictions.

Primary comparison: frozen source-only R6 versus frozen B1 on the same SciFact
examples and oracle evidence. Primary metric: three-label macro-F1. Secondary:
accuracy, helpful/harmful flips, route rate, forward passes, and the strict
unprojected result. Paired bootstrap and exact McNemar are required.

## Leakage boundary

SciFact dev labels are read only after B1, A3, memory, router, memory size, label
projection, and evaluation protocol are frozen. They may be used for final
metrics but not for model selection. Any later SciFact adaptation must be a
separate experiment with a new protocol and cannot reuse this result as an
untouched external evaluation.
