# Canonical R6-NLF ablation results

This report freezes the canonical-tokenizer Group-C rerun and direct memory
ablations for the final no-label-filter (NLF) policy. All results use grouped
AVeriTeC internal evaluation with fold-excluded source memory. They are
development evidence, not an independent test set.

## Group C: canonical design selection

| Variant | Natural routes | Natural Acc. | Natural Macro-F1 | C0-matched Acc. | C0-matched Macro-F1 |
|---|---:|---:|---:|---:|---:|
| C0 R6-HardLabel | 343 | 75.52 | 55.77 | 75.52 | 55.77 |
| C2-4 Hellinger | 327 | 75.65 | 56.21 | 75.59 | 56.13 |
| C4-1 Uniform weights | 353 | 75.39 | 56.28 | 75.39 | 56.45 |
| C5-2 Top-3 | 287 | 75.46 | 56.24 | 75.85 | 56.89 |
| C5-4 Top-20 | 217 | 75.62 | 55.21 | 76.01 | 55.76 |
| C7-1 R6-NLF | 375 | 75.85 | 56.66 | 75.75 | 56.55 |

All Holm-adjusted Accuracy comparisons against C0 are null. Top-3 and NLF
have positive matched Macro-F1 improvements when the reported C0-minus-variant
bootstrap intervals are sign-reversed, but this Macro-F1 family was not
Holm-adjusted. NLF remains the final policy because it gives the strongest
joint natural Accuracy/Macro-F1 result and was frozen before renewed external
evaluation. Top-3 and Top-20 are stronger at particular matched-budget
objectives, not at their natural operating points.

## Direct NLF memory ablation

All matched rows below route 375 examples (12.22%). B1 and B14 are 20-seed
means; paired tests use seed 2026.

| Variant | Accuracy | Macro-F1 | Interpretation |
|---|---:|---:|---|
| B0 No Memory | 74.02 | 54.60 | frozen verifier |
| B1 Random Memory | 73.89 | 54.79 | 20-seed mean |
| B14 Stratified Outcome Shuffle | 74.26 | 54.70 | 20-seed mean |
| B15 Class-Conditional Utility | 75.33 | 54.53 | no instance-level retrieval |
| B10 Recovery-Only Utility | 75.78 | 56.31 | no regression term |
| B9 Full Outcome Memory | **75.85** | **56.66** | final R6-NLF |

Key paired results, oriented as B9 minus control:

- B9 versus B1: Accuracy CI `[.0140,.0297]`, Macro-F1 CI
  `[.0100,.0377]`, Holm Accuracy `p=3.81e-7`.
- B9 versus B14: Accuracy `+1.86` points, Macro-F1 `+2.46` points;
  CIs `[.0111,.0261]` and `[.0140,.0359]`; Holm Accuracy
  `p=1.03e-5`.
- B9 versus B15: Accuracy `+0.52` points (Holm `p=.609`), Macro-F1
  `+2.13` points with CI `[.0085,.0351]`.
- B9 versus B10: Accuracy `+0.07` and Macro-F1 `+0.35` points; both
  paired intervals cross zero.
- B9 versus B16 uniform outcome weights: outputs are nearly identical;
  the current evidence does not show value from similarity weighting after
  top-k selection.

Natural B10 routes all 3,068 examples and produces 192 helpful versus 205
harmful changes (net `-13`). Natural B9 routes 375 and produces 94 helpful
versus 38 harmful changes (net `+56`). This supports the regression term as an
abstention mechanism even though fixed-budget quality is statistically tied.

## Budget interaction

Error-only memory leads full memory at 5% and marginally at 10%. Full memory
leads at 11.18%, ties at 15%, and is more accurate from 20% through 40%.
Recovery-only utility is strongest through 20%, then falls below full memory
from 25% onward. The B9--B7 Accuracy advantage is Holm-significant at 40%
(`p=.0412`), while several earlier budgets remain inconclusive.

## Integrity notes

- `compatibility_mode` is `none` in both memory-ablation summaries.
- The matched NLF route count is 375.
- Group C uses canonical `[\w'-]+` claim tokenization.
- Historical HardLabel component removals remain appendix-only and are not
  interpreted as direct proofs about final R6-NLF.
