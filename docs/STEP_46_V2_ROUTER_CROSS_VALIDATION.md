# Step 46 — Population-weighted leave-one-out router CV

## Why raw screening accuracy is invalid

The 155-example screen contains 115 baseline errors and only 40 correct
controls, while the full train baseline is 68.64% accurate. Optimizing raw screen
accuracy would reward excessive routing. Each screen group is therefore weighted
back to its observed frequency in all 3,068 train examples.

## Leakage control

For every query, all three action records belonging to that claim are excluded
from retrieval. Retrieval requires the same observable baseline prediction and
uses claim lexical similarity plus baseline label-score-distribution similarity.
Gold transition names are used only for population weighting and outcome scoring.

## Sweep

- Top-k memories per action: 3, 5, 10, 20.
- Minimum support: 2, 3, 5.
- Minimum utility: 0.00, 0.05, 0.10.
- Cost weight: 0.00, 0.05, 0.10.
- Uncertainty: full or one-minus baseline confidence.

Total: 216 configurations. Selection prioritizes population-weighted accuracy,
then macro-F1, then lower route rate.

## Result

Pending server execution. This remains train-only model selection; no holdout
labels or candidate outcomes are read.
