# P0 NB8 control and observed worked cases

## NB8 result

The grouped AVeriTeC rerun used 3,068 claims and the frozen NB0 natural budget
of 375 routes. NB8 retained NB0's exact global top-five support/denominator
and neutralized only cross-label recovery/regression outcomes.

| Router | Mode | Routes | Helpful | Harmful | Net | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| NB0 Full NLF | natural | 375 | 94 | 38 | 56 | .758475 | .566569 |
| NB8 same-label outcomes/global support | natural | 333 | 93 | 37 | 56 | .758475 | .565460 |
| NB0 Full NLF | matched | 375 | 94 | 38 | 56 | .758475 | .566569 |
| NB8 same-label outcomes/global support | matched | 375 | 100 | 45 | 55 | .758149 | .566287 |

For matched NB0 minus NB8, the paired accuracy interval is
[-.002934,.003585], the macro-F1 interval is [-.005405,.005382], and the
Holm-adjusted exact accuracy p-value is 1.0. This does not establish that
cross-label corrective outcomes cause NLF's gain. The supported positioning is
a compatibility relaxation with candidate-pool competition and selective
pruning at some operating points.

## Worked cases

The fold-safe exporter found 10 eligible harm-prevention cases and 11 eligible
helpful-rescue cases. Selection was post-hoc and gold-aware for explanation
only.

- `train:675`: gold/B1/A3 = REFUTED/REFUTED/CONFLICTING. HardLabel routes at
  utility +.0176 and causes a regression; NLF abstains at -.3391. Its global
  neighborhood contains two same-label neutral records and three cross-label
  records, two of which are harmful. NLF recovery/regression/cost estimates
  are .194/.433/1; HardLabel estimates are .314/.196/1.
- `train:1933`: gold/B1/A3 = NEI/REFUTED/NEI. HardLabel abstains at -.4585;
  NLF routes at +.0219 and recovers the example. Its neighborhood contains two
  helpful and two neutral cross-label records plus one harmful same-label
  record. NLF recovery/regression/cost estimates are .438/.316/1; HardLabel
  estimates are .202/.561/1.

These examples make the mechanism inspectable but do not override the null
aggregate NB8 and shuffle controls.
