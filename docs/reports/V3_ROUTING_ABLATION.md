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
