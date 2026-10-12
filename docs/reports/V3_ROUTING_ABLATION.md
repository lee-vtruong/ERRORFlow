# ERRORFlow V3 routing ablation

Status: **IMPLEMENTED; OUT-OF-FOLD RESULT PENDING**.

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
