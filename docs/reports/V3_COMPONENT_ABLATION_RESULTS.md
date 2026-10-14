# ERRORFlow V3 similarity-component ablation results

Status: **COMPLETE; PAPER INTEGRATED**.

Protocol: AVeriTeC grouped internal evaluation, 3,068 claims, token-Jaccard
0.90 grouped fold exclusion, fixed cached B1/A3, and 343 routes in matched
mode. Prediction replay checked 42,952 rows with zero mismatches.

## Main matched-budget result

| Variant | Accuracy | Macro-F1 | A0 minus variant Accuracy | A0 minus variant F1 | Holm p |
|---|---:|---:|---:|---:|---:|
| A0 Full R6 | .7552 | .5577 | -- | -- | -- |
| A1 w/o claim | .7546 | .5595 | +.0007 | -.0018 | 1.000 |
| A2 w/o label score | .7520 | .5640 | +.0033 | -.0063 | 1.000 |
| A3 w/o evidence length | .7533 | .5452 | +.0020 | +.0125 | 1.000 |
| A4 label score only | .7536 | .5506 | +.0016 | +.0071 | 1.000 |
| A5 claim only | .7451 | .5459 | +.0101 | +.0118 | .0511 |
| A6 evidence length only | .7389 | .5474 | +.0163 | +.0104 | .000150 |
| A7 uniform weights | .7539 | .5645 | +.0013 | -.0068 | 1.000 |
| A8 random neighbors, seed 2026 | .7448 | .5410 | +.0104 | +.0167 | .0336 |

Holm adjustment is over one family of eight A1--A8 exact McNemar tests. It
tests paired correctness, not macro-F1. The A8 20-seed mean is .7466 accuracy
and .5487 macro-F1; seed 2026 is the predeclared inferential comparison.

## Interpretation

- A6 and A8 provide corrected evidence that Full R6 accuracy benefits from a
  richer representation and similarity-based neighbor ranking.
- A5 is directionally worse but borderline after Holm (`p=.0511`).
- A3 has a positive A0-minus-variant macro-F1 interval, but no corrected
  accuracy difference.
- A1 does not establish an individual contribution from claim similarity.
- A2 and A7 achieve higher macro-F1 than A0, so neither label-score inclusion
  nor the manual 0.35/0.50/0.15 weights are established as uniformly better.

These negative findings are retained in the paper. They imply feature
redundancy and a quality trade-off, not permission to tune on external sets.

## A8 sensitivity

Across 20 seeds, A0 minus A8 matched-budget accuracy has mean .00859 and range
[.00522,.01499]. Macro-F1 has mean .00900 and range [.00061,.01671].

Server artifacts:

```text
outputs/v3/ablations/component_removal_grouped090/summary.json
outputs/v3/ablations/component_removal_grouped090/routes.jsonl
outputs/v3/ablations/component_removal_grouped090/predictions.jsonl
outputs/v3/ablations/component_removal_grouped090/paired_analysis/significance.json
outputs/v3/ablations/component_removal_grouped090/paired_analysis/paired_predictions.jsonl
```
