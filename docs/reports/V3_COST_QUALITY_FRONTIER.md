# ERRORFlow V3 cost-quality frontier

Status: **BOUNDED SCREEN OBSERVED; FULL-TRAIN CONFIRMATION PENDING**.

The 150-example action screen indicates that always applying
`conflicting_rescue_v1` provides the largest projected accuracy gain but doubles
verifier work from four to eight forward passes per claim. Confidence routing
retains most of that projected gain at lower cost:

- R3 at 20% routing: delta accuracy +0.03436, delta macro-F1 +0.01733,
  4.800 average forward passes.
- R4 at 35% routing: delta accuracy +0.03393, delta macro-F1 +0.02245,
  5.398 average forward passes.
- R1 always: delta accuracy +0.03840, delta macro-F1 +0.01731,
  8.000 average forward passes.

No operating point is frozen. Thresholds and budgets were inspected on the
same development screen, and the screen uses deterministic margin-spanning
sampling rather than random sampling. The next experiment evaluates cached A3
outcomes over all 3,068 train examples with grouped fold exclusion. Nested
inner-fold selection and independent external validation remain pending.

## Full-train A3 update

On all 3,068 train examples, R6 routes 11.64% of claims and reaches accuracy
delta +0.01891 and macro-F1 delta +0.01546 at 4.465 average forward passes.
R5 at 20% reaches +0.01956 accuracy and +0.02319 macro-F1 at 4.799 passes.
Always-on A3 costs 8.0 passes and decreases accuracy by 0.00424. This makes R6
the current best predeclared cost-aware operating point, subject to paired
uncertainty analysis and external validation.
