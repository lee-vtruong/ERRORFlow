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
