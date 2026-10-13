# ERRORFlow V3 cross-domain validation

Status: **PROTOCOL FROZEN; SCIFACT B1 AND A3 INFERENCE COMPLETE; ROUTING/EVALUATION PENDING**.

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

## Prepared external artifact

Official SciFact dev preparation completed with 300 unique claims:

- NOT ENOUGH INFO: 112;
- SUPPORTED: 124;
- REFUTED: 64;
- missing corpus documents: 0;
- invalid rationale sentence indices: 0.

The corpus SHA-256 is `b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62`;
the dev-claims SHA-256 is
`86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217`.
NEI examples have no official gold rationale and therefore carry empty oracle
evidence. This is dataset semantics, not a preprocessing failure.

## Frozen inference checkpoint

Both model passes completed over all 300 SciFact dev examples with the same
model, input artifact, decoding rule, label set, and seed. The input SHA-256
recorded by both manifests is
`c93d71875903d6fead298c2600cf706080a0a1d7fffb810617c5cd62b34e25bd`.

- B1 `conflict_audit_v1`: 300/300 rows, prompt SHA-256
  `700ebdc1b55cb9f04f7524f31b7a5d3bd41813f198e5629fe75ca551333decf7`,
  1,200 model forward passes, 39.63 seconds;
- A3 `conflicting_rescue_v1`: 300/300 rows, prompt SHA-256
  `09e187ec063b5e3d572bcb672430167456e046d5665c9c72adb447aeef597aaf`,
  1,200 model forward passes, 39.78 seconds.

Both manifests have `external_evaluation=true` and `status=complete`. No
SciFact outcome has yet been used in routing. The next operation applies the
already frozen R6 configuration using only the complete AVeriTeC source memory,
then performs the predeclared three-label projection and paired evaluation.
