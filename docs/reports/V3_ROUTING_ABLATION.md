# ERRORFlow V3 routing ablation

Status: **BOUNDED SCREEN COMPLETED; FULL-TRAIN FIXED-POLICY TEST PENDING**.

Implemented cached-action policies:

- R0 never route;
- R1 always route to a predeclared default action;
- R2 deterministic random routing;
- R3 confidence/margin-only routing;
- R4 prediction-label-conditional confidence routing;
- R5 memory-conditioned action routing;
- R6 cost-aware utility routing.

Budgets are population-weighted and capped at `0/5/10/20/35/50/100%`.
Each screen query excludes all memory records in its grouped outer fold. The
screen is reweighted to the full B1 train transition population so its
error-heavy construction cannot inflate gains.

The current evaluator is an outer-fold screening evaluator. Nested inner-fold
hyperparameter selection is explicitly **NOT IMPLEMENTED** yet and no result
from this command may be described as nested CV.

## Bounded-screen result

The first R0–R6 run used the 150-example transition-stratified action screen
and descriptive population weights. Key points on the cost-quality frontier:

| Policy | Budget | Route rate | Weighted net | Delta accuracy | Delta macro-F1 | Avg passes |
|---|---:|---:|---:|---:|---:|---:|
| R0 never | 0% | 0.000 | 0.0 | 0.00000 | 0.00000 | 4.000 |
| R1 always A3 | 100% | 1.000 | +117.8 | +0.03840 | +0.01731 | 8.000 |
| R3 confidence | 20% | 0.200 | +105.4 | +0.03436 | +0.01733 | 4.800 |
| R4 label-conditional | 35% | 0.350 | +104.1 | +0.03393 | +0.02245 | 5.398 |
| R5 memory | 20% | 0.200 | -98.4 | -0.03207 | -0.03578 | 4.799 |
| R6 utility | 20% | 0.200 | -98.4 | -0.03207 | -0.03578 | 4.799 |

R5 and R6 were harmful at every tested budget. R6 resembled R5 because all
actions have the same four-forward incremental cost, so the cost term rarely
changes their ordering. The current structured-memory hypothesis is therefore
unsupported by this screen, while simple confidence policies remain strong
comparators.

These are adaptive, train-side screen findings, not nested-CV or holdout
results. The next bounded experiment runs only the globally positive A3 action
on all 3,068 train examples. That removes unstable screen population weights
and tests whether fixed confidence and memory policies remain beneficial before
authorizing the other four full action runs.

## Full-train A3 fixed-policy result

The 3,068-example cached A3 experiment reverses the sparse-screen memory
finding. With same-outer-fold memories excluded, R5 at a fixed 20% matched
budget yields net +60, accuracy delta +0.01956, and macro-F1 delta +0.02319.
The cost-aware R6 policy routes only examples with estimated utility above zero:
11.64% of the population, net +58, accuracy delta +0.01891, macro-F1 delta
+0.01546, and 4.465 average forward passes.

Always applying A3 produces net -13 and accuracy delta -0.00424, despite a
macro-F1 delta of +0.00983. The selective gain therefore cannot be attributed
to A3 being uniformly stronger than B1. At comparable budgets, memory routing
also exceeds deterministic random, confidence-only, and label-conditional
routing in both accuracy and macro-F1.

R6 with `top_k=5`, `min_support=2`, `cost_weight=0.1`, and utility threshold
`>0` is retained as the fixed primary V3 candidate because these settings were
specified before inspecting the full-train curve. R5 at 20% is an exploratory
upper operating point, not a frozen primary result. Per-example predictions
were materialized for the paired analysis below.

## Paired statistical analysis

The materialized fixed R6 policy improves B1 from 0.74022 to 0.75913 accuracy
and from 0.54602 to 0.56148 macro-F1. Of 140 discordant examples, R6 alone is
correct on 99 and B1 alone is correct on 41, for net +58. Exact two-sided
McNemar `p=1.03e-6`.

With 5,000 paired bootstrap samples, the 95% interval for accuracy delta is
`[0.01108, 0.02673]`; the macro-F1 delta interval is
`[0.00346, 0.02745]`. The empirical probabilities of positive deltas are
1.000 and 0.995 respectively. Both intervals exclude zero.

This supports a train-internal, fold-excluded benefit over B1. It does not
remove adaptivity from earlier action screening, replace nested inner-fold
selection, or establish independent-domain generalization.

## Fold and class stability

R6 routes 357/3,068 examples (11.64%), with 99 helpful and 41 harmful flips.
All five grouped outer folds improve in both metrics. Fold accuracy deltas span
`[+0.01305, +0.02769]`; fold macro-F1 deltas span
`[+0.00503, +0.03458]`. Route rates remain between 10.44% and 12.72%, and every
fold has positive net correction.

Per-class F1 changes relative to B1 are:

| Label | B1 F1 | R6 F1 | Delta |
|---|---:|---:|---:|
| CONFLICTING | 0.08230 | 0.08730 | +0.00500 |
| NOT ENOUGH INFO | 0.53571 | 0.56075 | +0.02503 |
| REFUTED | 0.81631 | 0.83813 | +0.02182 |
| SUPPORTED | 0.74975 | 0.75972 | +0.00998 |

The gain is broad across folds but not a substantive solution to CONFLICTING:
only 11/195 CONFLICTING examples are correct under R6 versus 10/195 under B1.
The defensible description is selective residual correction, not conflicting-
evidence resolution.
