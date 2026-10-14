# ERRORFlow V3 Group-C Interpretation

## Decision

Do not replace the frozen R6 main result with a Group-C winner on the basis of
this development ablation alone. Keep C0/R6 as the confirmatory method in the
current paper and report C5-2 Top-3 and C7-1 No Label Filter as promising
post-hoc development variants.

If a successor is to be frozen for a genuinely untouched evaluation, prefer
No Label Filter as the primary candidate. It improves both natural-threshold
and matched-budget results, whereas Top-3's strongest result occurs under the
forced matched budget.

## Key matched-budget results

All methods route 347 of 3,068 claims (11.31%).

| Variant | Accuracy | Macro-F1 | Helpful | Harmful | Net |
|---|---:|---:|---:|---:|---:|
| C0 Full R6 | 75.52 | 55.67 | 90 | 44 | 46 |
| C5-2 Top-3 | 75.78 | 56.86 | 103 | 49 | 54 |
| C7-1 No Label Filter | 75.78 | 56.52 | 88 | 34 | 54 |

Relative to C0, Top-3 changes Accuracy by +0.26 percentage points and Macro-F1
by +1.19 points. No Label Filter changes Accuracy by +0.26 points and Macro-F1
by +0.85 points.

The paired bootstrap intervals are stored as C0 minus candidate. Therefore:

- Top-3 Macro-F1: C0-minus-Top-3 CI `[-2.16, -0.27]` points.
- No Label Filter Macro-F1: C0-minus-No-Filter CI `[-1.64, -0.14]` points.
- Both Accuracy intervals cross zero.
- Exact McNemar Accuracy p-values are 0.389 for Top-3 and 0.243 for No Label
  Filter; Holm-adjusted Accuracy p-values are 1.0.

The positive unadjusted Macro-F1 evidence is useful but is not yet a
multiplicity-corrected confirmatory result, because Group C inspected more than
30 development variants and the current Holm family covers Accuracy only.

## Natural-threshold behavior

| Variant | Route rate | Accuracy | Macro-F1 | Helpful | Harmful | Net |
|---|---:|---:|---:|---:|---:|---:|
| C0 Full R6 | 11.31 | 75.52 | 55.67 | 90 | 44 | 46 |
| C5-2 Top-3 | 9.58 | 75.42 | 56.23 | 83 | 40 | 43 |
| C7-1 No Label Filter | 12.58 | 75.95 | 56.70 | 96 | 37 | 59 |

Top-3 trades a slightly lower Accuracy and lower intervention rate for better
Macro-F1. No Label Filter is the more consistent development candidate: its
natural policy improves both metrics and its gain is associated with fewer
harmful corrections despite routing more claims.

## Mechanistic interpretation

The hard same-predicted-label compatibility rule appears unnecessarily
restrictive. Historical correction outcomes from other predicted-label groups
can be informative when claim, score-profile, and evidence features remain in
the retrieval score. Removing the hard filter increases the natural net gain
from 46 to 59.

Top-3 suggests that very local evidence may improve class balance, but the
matched-budget improvement partly comes from forcing 347 routes even though
its natural utility rule selects only 294. This makes Top-3 less compelling as
an immediate drop-in replacement unless the route budget is explicitly part of
the method definition.

## P2 findings

Neither source-fold tuned weights nor the trained local logistic comparator
provides a confirmed improvement over C0. Their Holm-adjusted Accuracy p-values
are approximately 0.904, and their bootstrap intervals include no difference.
This does not justify replacing the training-free router with a trained router.

## Research-integrity constraint

SciFact and CLIMATE-FEVER results have already been opened. Re-running a newly
selected Group-C winner on those datasets can be reported as post-hoc transfer
analysis, but not as an untouched confirmatory test. A method change requires
one of the following before replacing the main claim:

1. freeze the candidate and evaluate on a new untouched dataset or split; or
2. perform nested source-only variant selection, choosing the Group-C variant
   on inner folds and scoring it on the corresponding held-out outer fold.

Do not combine Top-3 and No Label Filter after observing these results and call
the combination confirmatory. Such a combination is a new post-hoc candidate.

## Paper treatment

- Keep C0/R6 for all existing main and external results.
- Add Top-3 and No Label Filter to the main ablation table or a compact
  development-sensitivity table.
- State that hard label compatibility is not empirically necessary under this
  development protocol.
- Describe No Label Filter as a candidate design revision, not the reported
  ERRORFlow main method.
