# V3 memory-ablation paper integration

Status: integrated into `paper/errorflow_v3.tex`; PDF render intentionally
deferred.

Source artifact supplied after the server run:
`MEMORY_ABLATION_B0_B17_RESULTS.txt`.

## Main-paper configurations

All routed configurations use 343 claims (11.18%) unless noted. B1 and B14
are 20-seed means; their significance tests use the predeclared seed 2026.

| ID | Configuration | Accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|
| B0 | No memory / B1 only | 74.02 | 54.60 |
| B1 | Random compatible memory | 74.66 | 54.87 |
| B14 | Stratified outcome shuffle | 74.34 | 54.51 |
| B15 | Class-conditional utility | 75.42 | 54.63 |
| B10 | Recovery-only utility | 75.62 | 55.63 |
| B9 | Full outcome memory | 75.52 | 55.77 |

The primary seed-2026 B9--B14 comparison is +1.4016 accuracy points and
+1.7078 macro-F1 points; exact McNemar `p=0.0003682`, Holm-adjusted
`p=0.0025772`. B9--B10 and B9--B15 do not have Holm-significant correctness
differences.

## Appendix configurations

| ID | Configuration | Routes | Accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|
| B2 | Claim-nearest | 343 | 74.51 | 54.59 |
| B3 | Score-nearest | 343 | 75.36 | 55.06 |
| B7 | Error-only | 343 | 75.72 | 56.28 |
| B8 | Correct-only | 343 | 73.83 | 54.44 |
| B11 | Regression-only | 343 | 73.92 | 54.48 |
| B12 | Size-matched full, 20-seed mean | 343 | 75.16 | 55.31 |
| B13 | Size-matched mixed, 20-seed mean | 343 | 75.12 | 55.37 |
| B16 | Uniform outcome weights | 343 | 75.49 | 55.75 |
| B17 | Nested calibrated utility | 382 | 75.49 | 55.94 |

B4/B14 and B5/B15 were verified as implementation-equivalent through their
shared code paths and identical outputs. B6 is a constant global prior and
remains artifact-only.

## Budget finding

Error-only memory has higher point-estimate accuracy at 5%, 10%, and 11.18%.
Full memory has higher point-estimate accuracy at every tested budget from 15%
through 40%. The paired B9--B7 accuracy intervals cross zero at all reported
budgets, so the paper describes a budget-dependent trade-off rather than a
confirmed universal ordering.

Added figure: `paper/figures/memory_budget_tradeoff.pdf`. It can be reproduced
from the server summary with `scripts/plot_v3_memory_budget.py`.

## Claim boundaries

- Outcome shuffling supports the value of feature-conditioned historical
  outcomes.
- The results do not establish that the regression term improves quality at a
  fixed route count.
- B10 natural routing invokes A3 for all 3,068 claims; regression modeling is
  therefore discussed as a selective-coverage mechanism.
- B7 is reported despite outperforming B9 at the primary matched budget.
- B17 is not presented as a matched-budget comparison because it routes 382
  claims.
