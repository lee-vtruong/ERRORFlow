# ERRORFlow V3 matched-budget routing ablation

Status: **COMPLETE**.

## Confirmatory question

Does structured residual memory add correction value beyond uncertainty,
label-conditional, similarity-only, static-pattern, and random routing when all
methods receive the same opportunity to invoke cached A3?

## Systems

- `B1`: no correction;
- `A3-All`: correction on every example;
- `R-Random`: 20 deterministic seeds;
- `R-Conf`: lowest maximum B1 label probability;
- `R-Margin`: smallest B1 top-two probability margin;
- `R-Label`: margin routing separately within each predicted label;
- `R-Sim`: highest similarity to compatible source memory, ignoring outcomes;
- `R-Static`: utility of a coarse B1-label × evidence-count rule;
- `R-Memory`: local recovery-minus-regression without the cost penalty;
- `R6`: local recovery-minus-regression-minus-cost, restricted to positive
  estimated utility.

The evaluator uses the conservative token-Jaccard 0.90 connected-component
folds established after WP0. Every memory-derived score excludes the complete
query fold. The A3 prediction cache is shared by every router.

## Budgets and measurements

Requested route budgets are `0/5/10/11.64/15/20/35/50/100%`, plus the exact
natural positive-utility R6 rate induced by the conservative folds. R6 may
route fewer examples than a requested budget because it never admits
non-positive utility. Random routing reports 20 seeds and empirical seed
variation.

Metrics include accuracy, macro-F1, per-class F1, helpful/harmful/net flips,
forward passes, generated tokens, and cached-component mean/p95 latency.
Tokenizer-level input tokens were not recorded historically; the evaluator
reports reprocessed claim/evidence whitespace tokens separately and leaves
model tokenizer input tokens null rather than fabricating them.

Primary comparison is R6 against the strongest deterministic non-memory router
at the natural R6 route count. Exact McNemar, paired bootstrap, and Holm-adjusted
McNemar p-values are produced. Because A3 was selected on an AVeriTeC train
screen, this remains `VERIFIED_BUT_TRAIN_SELECTED` evidence even with strict
fold isolation.

## Results

The conservative token-Jaccard 0.90 grouping changed the natural R6 budget to
`343/3068 = 11.18%`. Results at that exactly matched route budget are:

| Router | Accuracy | Macro-F1 | Helpful | Harmful | Net | Mean passes |
|---|---:|---:|---:|---:|---:|---:|
| B1 | 0.7402 | 0.5460 | 0 | 0 | 0 | 4.000 |
| A3-All | 0.7360 | 0.5559 | 192 | 205 | -13 | 8.000 |
| R-Random (20-seed mean) | 0.7393 | 0.5465 | 21.8 | 24.4 | -2.7 | 4.447 |
| R-Conf | 0.7392 | 0.5458 | 60 | 63 | -3 | 4.447 |
| R-Margin | 0.7454 | 0.5522 | 68 | 52 | 16 | 4.447 |
| R-Label | 0.7451 | 0.5547 | 71 | 56 | 15 | 4.447 |
| R-Sim | 0.7331 | 0.5416 | 11 | 33 | -22 | 4.447 |
| R-Static | 0.7526 | 0.5451 | 60 | 22 | 38 | 4.447 |
| R-Memory | **0.7552** | **0.5577** | **90** | **44** | **46** | 4.447 |
| R6 | **0.7552** | **0.5577** | **90** | **44** | **46** | 4.447 |

Relative to B1, R6 improves accuracy by `+0.01499` and macro-F1 by
`+0.01169`. Exact McNemar gives `p=8.75e-5` and Holm-adjusted
`p=6.12e-4`; the paired 95% intervals are `[0.00750, 0.02249]` for accuracy
and `[0.00009, 0.02393]` for macro-F1. R6 also exceeds the random-router mean
by `+0.01587` accuracy, `+0.01117` macro-F1, and `+48.7` net corrections.

The strongest deterministic non-memory router under the primary macro-F1
criterion is R-Label. R6 is `+0.01010` in accuracy and `+0.00298` in macro-F1
above R-Label. Its accuracy bootstrap interval is positive, but the macro-F1
interval crosses zero and the Holm-adjusted McNemar result is `p=0.0674`.
Therefore the incremental advantage over the strongest simple non-memory
router is promising but not confirmatory.

R-Static has nearly the same accuracy as R6 (`0.7526` versus `0.7552`) but
substantially lower macro-F1 (`0.5451` versus `0.5577`). The macro-F1 paired
bootstrap interval for R6 minus R-Static is positive, while the correctness
comparison is not significant (`Holm p=0.9935`). These statistics answer
different questions and should be reported together.

## WP1 gate decision

**Partial support, with a required claim correction.** Structured outcome
memory is useful relative to random, confidence-only, similarity-only, and B1.
However, this experiment does not establish a statistically secure macro-F1
advantage over the strongest simple label-conditional router after multiple
comparison correction.

Most importantly, R-Memory and R6 select exactly the same 343 examples and
produce identical predictions. With one corrective action and a constant
four-forward-pass action cost, subtracting the same cost from every memory
utility cannot change the ranking at a fixed budget. The validated mechanism
is therefore **residual outcome-memory routing**, not an incremental benefit
from cost-aware scoring. A distinct cost-aware claim requires heterogeneous
actions/costs or another design in which cost changes route ordering.

WP1 is closed as a qualified positive result. The paper-facing claim should be
restricted to selective memory-conditioned correction under a fixed action,
while retaining the train-selected limitation from WP0.
