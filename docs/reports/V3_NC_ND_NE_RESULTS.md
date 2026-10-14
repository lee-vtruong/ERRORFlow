# ERRORFlow V3 final-NLF ablation results

Source: grouped AVeriTeC internal evaluation, 3,068 claims, duplicate-group
threshold 0.90, five fold-excluded source-memory folds, seed 2026. B1 accuracy
is 74.02% and Macro-F1 is 54.60%. Canonical R6-NLF routes 375 claims (12.22%)
and reaches 75.85% accuracy and 56.66% Macro-F1 (94 helpful, 38 harmful,
net +56).

## N-C: similarity components

All values below use the matched 375-route budget. Holm correction covers
NC1--NC8.

| ID | Configuration | Accuracy | Macro-F1 | Holm p (Acc.) |
|---|---|---:|---:|---:|
| NC0 | Full NLF | 75.85 | 56.66 | -- |
| NC1 | w/o claim Jaccard | 75.52 | 56.12 | .486 |
| NC2 | w/o label-score similarity | 74.28 | 55.68 | 4.30e-4 |
| NC3 | w/o evidence length | 75.33 | 54.96 | .400 |
| NC4 | Uniform component weights | 75.26 | 56.67 | .164 |
| NC5 | Label-score only | 75.20 | 55.08 | .299 |
| NC6 | Hellinger | 75.13 | 55.96 | .019 |
| NC7 | Jensen--Shannon | 74.58 | 55.94 | .00125 |
| NC8 | TF-IDF claim similarity | 75.55 | 56.17 | .486 |

Label-score similarity has the strongest corrected accuracy evidence. The
NC0-minus-NC3 Macro-F1 interval is positive, supporting evidence length for
class balance. Claim Jaccard and manual component weights are not individually
established.

## N-D: utility and decisions

| ID | Configuration | Natural routes | Natural Acc. | Matched Acc. | Matched F1 |
|---|---|---:|---:|---:|---:|
| ND0 | Full signed utility | 375 | 75.85 | 75.85 | 56.66 |
| ND1 | Recovery only | 3,068 | 73.60 | 75.78 | 56.31 |
| ND2 | Unsmoothed | 375 | 75.85 | 75.85 | 56.66 |
| ND3 | Source-prior shrinkage | 375 | 75.85 | 75.85 | 56.66 |
| ND4 | Regression only | 3,068 | 73.60 | 73.99 | 54.62 |
| ND6 | Nested calibrated threshold | 393 | 75.85 | -- | 56.74 |

ND2 and ND3 reproduce ND0 routes and final predictions. ND2 changes 933
utility ranks (maximum absolute rank change 350), so equality is specific to
the decision boundary. ND4 is significantly worse than ND0 after Holm
correction (p=4.88e-6). Recovery-only is indistinguishable at matched budget
but routes every claim naturally and yields net -13.

The utility threshold is locally flat: -0.05 and 0 both route 375 claims.
Threshold -0.10 routes 452 and reaches 75.88/56.72; it is sensitivity, not a
new externally selected policy.

## N-E: sensitivity and robustness

- Top-k 1/3/5/10/20 gives accuracy 74.93/75.49/75.85/75.42/75.36 and
  Macro-F1 55.87/56.30/56.66/55.91/55.78.
- Minimum support 1/2/3/5 is invariant at this setting because every selected
  top-five neighborhood has full support.
- Exact budgets 5/10/12.22/15/20/30/40% give accuracy
  74.93/75.52/75.85/75.88/75.88/75.55/75.55.
- Twenty-seed mean accuracy rises from 74.09 at 25 source records to 75.16 at
  1,000; full memory reaches 75.85. Mean Macro-F1 rises from 54.74 to 55.98,
  then 56.66 with full memory.
- Group thresholds .85/.90/.95 give accuracy 75.59/75.85/75.65 and Macro-F1
  56.39/56.66/56.15.
- Empirical net correction rates are negative for utility below -0.05 and
  positive in the observed positive-utility bins; sparse near-zero bins prevent
  a strong calibration claim.

## NE6: external source-memory composition

NE6 is post-hoc sensitivity, not model selection. Restricting source memory
changes natural route rates from 0% to 85%, so subset quality is not a
matched-budget causal comparison. Full-memory R6-NLF obtains 53.68/41.17 on
CLIMATE-FEVER, 63.00/59.51 on SciFact oracle, and 63.00/60.75 on SciFact BM25
(accuracy/Macro-F1). Some label subsets have much higher point estimates---for
example source-predicted-CONFLICTING reaches 77.33/74.66 on SciFact oracle---
but routes 59.7% rather than 8.0%. The best subset varies by evidence protocol.
No target-specific memory subset is promoted. Topic sensitivity is unavailable
because verified source topic metadata is absent.
