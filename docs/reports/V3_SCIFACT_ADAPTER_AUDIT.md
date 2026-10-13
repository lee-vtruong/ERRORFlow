# ERRORFlow V3 SciFact label-adapter audit

Status: **IMPLEMENTED; SERVER CONTROLLED EVALUATION PENDING**.

## Research question

Does the frozen source-only ERRORFlow improvement on SciFact survive a fully
transparent label protocol, and how much of the observed gain is created by
mapping AVeriTeC `CONFLICTING` predictions to SciFact `NOT ENOUGH INFO`?

## Adapter provenance

The projection implementation was committed in
`b2899476e3285202b665c14bd64e2c4e4bb67871` before the completed SciFact
inference/result commits. The frozen mapping is:

```text
CONFLICTING -> NOT ENOUGH INFO
```

Both B1 and R6 receive the identical mapping. The source-memory retrieval,
utility estimation, and route decision operate on raw four-label B1 scores.
Projection is applied only to final prediction files after routing and action
selection. Therefore projection cannot influence which examples are routed.

Historical artifacts already establish the central sensitivity:

- raw four-label R6 versus B1: `-0.0100` accuracy, `+0.00135` macro-F1,
  McNemar `p=0.6636`;
- projected three-label R6 versus B1: `+0.0300` accuracy, `+0.04563`
  macro-F1, McNemar `p=0.02246`;
- projection changes 5 B1 predictions and 12 R6 predictions;
- all 12 projected R6 `CONFLICTING` predictions become correct SciFact NEI
  predictions.

The audit script recomputes these facts from prediction-level artifacts and
writes one transition record for every claim whose raw or projected verdict
changes.

## Controlled systems

The controlled table includes:

1. B1 raw four-label behavior;
2. always-on A3 raw behavior;
3. R6 memory route decisions with no A3 prediction replacement;
4. full raw R6;
5. the same four systems after identical final-label projection;
6. direct three-label B1 scoring;
7. direct three-label always-on A3 scoring.

The memory-without-A3 control must equal B1 in verdict quality by construction:
retrieval and routing alone do not modify predictions. Always-on A3 is the
control with A3 but without utility-based memory selection.

Direct three-label scoring uses an ontology-specific prompt frozen in code
before execution. Mixed support/refutation maps to NEI because SciFact has no
conflict verdict. It is a separate sensitivity and is not claimed to be
mathematically equivalent to projecting four-label probabilities. Direct
three-label outputs are not routed through the four-label AVeriTeC memory,
because doing so would mix incompatible score vectors and prompt semantics.

## Leakage boundary

SciFact dev gold labels are used only by the post-inference audit. They do not
modify prompts, projection, source memory, routing hyperparameters, or model
weights. If this audit motivates a new adapter, that adapter cannot be claimed
as independently validated on the same 300 examples.

## Expected artifacts

```text
outputs/v3/scifact/adapter_audit/summary.json
outputs/v3/scifact/adapter_audit/transitions.jsonl
outputs/v3/scifact/direct3/conflict_audit_v1.jsonl
outputs/v3/scifact/direct3/conflicting_rescue_v1.jsonl
```

The final report section will be completed only from these real server
artifacts. Until then the historical projected transfer remains conditionally
supported and projection-dependent.
