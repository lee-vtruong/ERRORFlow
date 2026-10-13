# ERRORFlow V3 current research status

Status: **V3 SOURCE EVALUATION AND FIRST FROZEN CROSS-DOMAIN VALIDATION COMPLETE**.

## Executive conclusion

The evidence supports ERRORFlow as a training-free, selective residual
correction method, but with a narrower claim than universal error correction.
On AVeriTeC, source-only residual memory identifies a small subset on which a
second conflict-rescue inference is useful. On SciFact, the complete frozen
system transfers significantly after the predeclared target-label projection.
The raw four-label action does not significantly improve SciFact, so the
external result is conditional on label-space reconciliation.

`Training-free` here means that model weights are never updated: there is no
gradient descent, fine-tuning, LoRA, or adapter training. It does not mean
supervision-free. Gold-labelled AVeriTeC outcomes are used offline to construct
a non-parametric error memory and estimate action utility.

## Method

### B1 base verifier

- Model: Qwen3-4B-Instruct-2507.
- Input: claim plus oracle evidence.
- Prompt: immutable `conflict_audit_v1`.
- Decision: mean label log-probability over `SUPPORTED`, `REFUTED`,
  `NOT ENOUGH INFO`, and `CONFLICTING`.
- Cost: four model forward passes per example.

B1 is already a stronger conflict-first training-free prompt, not a trained
classifier and not the historical plain S0 prompt.

### ERRORFlow after B1

1. Run B1 and retain its prediction, label-score profile, confidence, and
   observable claim/evidence features.
2. Retrieve compatible source examples from the structured residual memory.
   Retrieval does not inspect the target gold label.
3. Estimate the utility of the fixed `conflicting_rescue_v1` action from its
   past helpful, harmful, and cost outcomes.
4. R6 routes only when estimated utility is greater than zero, using
   `top_k=5`, `min_support=2`, and `cost_weight=0.1`.
5. Routed examples use A3; non-routed examples retain B1. A routed example
   costs four additional model forward passes.
6. When target label spaces differ, apply only the adapter declared before
   reading target outcomes.

For source evaluation, retrieval excludes the query example and its grouped
outer fold. For SciFact, the complete 3,068-record AVeriTeC memory is frozen and
used source-only; no SciFact outcome updates the prompts, memory, or router.

## AVeriTeC results

### Base progression

| Verifier | Accuracy | Macro-F1 | Correct | Residual errors |
|---|---:|---:|---:|---:|
| Historical S0 | 0.68644 | 0.50691 | 2,106 | 962 |
| V3 B1 conflict-first | 0.74022 | 0.54602 | 2,271 | 797 |

The B1-versus-S0 gain is descriptive on AVeriTeC train, not an independent
generalization result.

### Frozen R6 versus B1

| Metric | B1 | R6 ERRORFlow | Delta |
|---|---:|---:|---:|
| Accuracy | 0.74022 | 0.75913 | +0.01890 |
| Macro-F1 | 0.54602 | 0.56148 | +0.01546 |
| Average forward passes | 4.000 | 4.465 | +0.465 |

R6 routes 357/3,068 examples (11.64%). It produces 99 helpful and 41 harmful
flips, for net +58 correct predictions. Exact McNemar gives `p=1.03e-6`.
Paired-bootstrap 95% intervals are `[0.01108, 0.02673]` for accuracy delta and
`[0.00346, 0.02745]` for macro-F1 delta. All five grouped outer folds improve
both metrics.

Always applying A3 is not beneficial: it changes accuracy by -0.00424 while
doubling work to eight forward passes. The gain therefore comes from selective
routing, not from replacing B1 with a uniformly stronger prompt.

The method improves all four per-class F1 values, but it does not solve the
minority `CONFLICTING` class: R6 gets only 11/195 correct versus 10/195 for B1.

### Memory-size result

| Memory size | Route rate | Net | Delta accuracy | Delta macro-F1 |
|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0 | 0.00000 | 0.00000 |
| 25 | 0.1102 | +39 | +0.01271 | -0.00055 |
| 50 | 0.1848 | +51 | +0.01662 | +0.01094 |
| 100 | 0.1340 | +63 | +0.02053 | +0.01374 |
| 200 | 0.1675 | +52 | +0.01695 | +0.00720 |
| 3,068 (frozen) | 0.1164 | +58 | +0.01890 | +0.01546 |

The curve is non-monotonic. The full memory is retained because macro-F1 is the
predeclared primary metric; 100 records is only an exploratory accuracy option.

## Frozen SciFact transfer

The external experiment uses all 300 official SciFact dev claims with official
gold-rationale oracle evidence. It is a verifier-transfer experiment, not an
open-retrieval or official end-to-end SciFact score.

SciFact has three labels, so the predeclared adapter maps `CONFLICTING` to
`NOT ENOUGH INFO` for the primary comparison.

| Metric | B1 projected | R6 projected | Delta |
|---|---:|---:|---:|
| Accuracy | 0.60667 | 0.63667 | +0.03000 |
| Macro-F1 | 0.55957 | 0.60520 | +0.04563 |
| Average forward passes | 4.000 | 4.453 | +0.453 |

R6 routes 34/300 examples (11.33%). The projected comparison has 11
candidate-only-correct and 2 baseline-only-correct examples. Exact McNemar is
`p=0.02246`; bootstrap 95% intervals are `[0.0067, 0.0533]` for accuracy delta
and `[0.0172, 0.0774]` for macro-F1 delta.

The adapter changes 5 B1 predictions and 12 R6 predictions. None of the five B1
changes becomes correct; all 12 R6 changes become correct NEI predictions.
Consequently, the label adapter is a central mechanism of the external gain.

The mandatory strict four-label sensitivity analysis is null: accuracy changes
by -0.0100 and macro-F1 by +0.00135, with McNemar `p=0.66362`; both bootstrap
intervals cross zero. ERRORFlow must therefore be described as transferring as
a complete frozen system, not as an adapter-independent action improvement.

## Hypothesis assessment

| Hypothesis | Assessment | Evidence |
|---|---|---|
| Conflict-first B1 is stronger than historical S0 | Descriptively supported | +5.38 accuracy and +3.91 macro-F1 points on AVeriTeC train |
| A3 is uniformly better than B1 | Rejected | Always-on A3 decreases full-train accuracy and doubles inference work |
| Error-memory selective routing improves B1 | Supported internally | Significant paired gains, positive result in all 5 folds |
| Selective routing is cost-efficient | Supported | About 11.6% routed for +1.89 accuracy and +1.55 macro-F1 points |
| The complete frozen system transfers to SciFact | Conditionally supported | Significant projected gains with source-only memory |
| The raw action transfers without label adaptation | Not supported | Raw sensitivity is statistically null |
| V3 solves conflicting-evidence classification | Not supported | Only 11/195 AVeriTeC `CONFLICTING` examples correct |

## What can be claimed now

The defensible claim is that a frozen, non-parametric error memory can support
cost-aware selection of a second inference action without updating model
weights, improving a strong verifier internally and transferring as a complete
system to a second dataset under an explicitly frozen label adapter.

The current evidence does not establish state of the art, an end-to-end fact-
checking gain, label-free learning, or universal cross-domain correction.

## Required next evidence

1. **Completed:** CLIMATE-FEVER provides a second locked external dataset with
   an ontology-aligned four-label protocol and official retrieved candidates.
2. **Completed:** SciFact open-corpus BM25 retrieval replaces oracle rationales
   and measures retrieval/action interactions without target tuning.
3. Use nested inner-fold policy selection if claiming a generally selected
   router rather than the current frozen configuration.
4. Compare against confidence-only and random routing under identical realized
   compute budgets in each new domain.
5. Treat the opened AVeriTeC dev labels as development evidence, not as a new
   untouched holdout.

## Retrieved-evidence update

The frozen SciFact BM25 condition retrieves five sentences from the complete
5,183-document corpus without cited-document or gold-rationale access. Top-5
any-rationale recall is 64.36% and complete-rationale recall is 62.77% on 188
verifiable claims.

Projected B1 scores 61.00% accuracy and 57.78 macro-F1. R6 routes 18.67% and
scores 62.67%/60.35, while A3-All scores 63.67%/61.57. R6's accuracy gain is
not significant, but its macro-F1 bootstrap interval over B1 is positive.
Neither R6 nor A3 is significantly superior to the other, and R6 is not
significantly above matched-route confidence or similarity controls.

The retrieved condition sharply reduces the oracle A3 result, while B1 and R6
remain nearly stable. Only five A3 outputs require the SciFact projection,
versus 96 under oracle evidence. This confirms that the earlier very large A3
oracle gain was mediated by both evidence quality and ontology projection;
the smaller retrieved-evidence result is the more realistic robustness signal.

Retrieval-stratified analysis locates the mechanism more precisely. A3-All
gains 15 net correct predictions on 112 NEI claims, but loses one on 121
verifiable retrieval hits and six on 67 retrieval misses. R6 gains five on NEI
and has net zero in both verifiable strata. On retrieval misses, R6 preserves
B1 accuracy while A3-All and confidence routing regress. The current evidence
therefore supports selective harm avoidance under missing evidence, not
recovery of facts that retrieval failed to supply.
