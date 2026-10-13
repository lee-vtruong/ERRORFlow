# ERRORFlow V3 SciFact label-adapter audit

Status: **COMPLETE**.

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

## Controlled evaluation result

All 300 official SciFact dev claims were evaluated with oracle rationale
evidence. The observed quality/cost results are:

| System | Label behavior | Accuracy | Macro-F1 | Mean passes |
|---|---|---:|---:|---:|
| B1 | raw four-label | 0.6067 | 0.4239 | 4.000 |
| A3-All | raw four-label | 0.5167 | 0.4208 | 8.000 |
| R6 memory, no A3 | raw four-label | 0.6067 | 0.4239 | 4.000 |
| R6 full | raw four-label | 0.5967 | 0.4253 | 4.453 |
| B1 | projected three-label | 0.6067 | 0.5596 | 4.000 |
| A3-All | projected three-label | **0.8167** | **0.7822** | 8.000 |
| R6 memory, no A3 | projected three-label | 0.6067 | 0.5596 | 4.000 |
| R6 full | projected three-label | 0.6367 | 0.6052 | 4.453 |
| B1 | direct three-label | 0.4900 | 0.3767 | 3.000 |
| A3-All | direct three-label | 0.4400 | 0.3051 | 6.000 |

R6 routes 34/300 examples. Under raw labels it is one accuracy point worse
than B1 and statistically unsupported, reproducing the historical sensitivity.
Under final-label projection it is three accuracy points and 4.56 macro-F1
points above B1.

Projection changes five B1 predictions but produces no B1 correctness gain.
It changes twelve R6 predictions and all twelve become correct NEI verdicts.
Thus the complete positive R6 delta is adapter-mediated rather than a raw
four-label correction effect.

The newly exposed decisive control is projected A3-All: it reaches 81.67%
accuracy and 78.22 macro-F1, far above projected R6. Raw A3-All is poor, so the
quality gain is specifically produced by interpreting A3 `CONFLICTING` outputs
as target-domain NEI. R6's AVeriTeC source-memory router invokes this useful
target behavior on only a small subset and therefore does not establish that
source-domain route utility transfers well.

Direct three-label scoring is also poor and nearly eliminates NEI recall
(B1: 0.0357; A3: 0.0089). Because this prompt protocol was frozen before its
result was observed, it is retained as a negative sensitivity and will not be
tuned on the same SciFact dev set.

Raw four-label macro-F1 and adapted three-label macro-F1 use different label
sets and must not be compared as if their denominators were identical. Within
each protocol, however, paired system comparisons remain valid.

## WP2 gate decision

The cross-domain claim is narrowed to the following:

> A predeclared final-label projection makes the frozen four-label A3 critic
> highly effective on SciFact oracle evidence; selective R6 preserves a small
> part of that gain at lower inference cost.

The data do **not** support unconditional raw-label transfer, direct three-label
robustness, or successful transfer of the AVeriTeC source-memory routing policy.
SciFact therefore supplies evidence for an ontology-adapted critic and a
quality/cost trade-off, not clean source-memory generalization.

Enhanced accounting confirms that projection changes 96 A3-All outputs,
producing 90 correctness gains and zero losses. Projected A3-All beats projected
B1 with 74 candidate-only versus 11 baseline-only correct examples (exact
McNemar `p=1.29e-12`); the paired 95% intervals are `[0.1533, 0.2634]` for
accuracy and `[0.1659, 0.2813]` for macro-F1. Projected R6 is substantially
worse than projected A3-All: 9 candidate-only versus 63 baseline-only correct
examples (`p=4.18e-11`), with accuracy and macro-F1 intervals entirely below
zero. These results close WP2 and make the projection dependence explicit.
