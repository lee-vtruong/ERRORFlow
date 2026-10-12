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

Completed on the server. The best configuration uses three memories per action,
minimum support two, minimum utility zero, cost weight 0.10, and inverse
confidence uncertainty. Population-weighted estimates are:

- Baseline accuracy: 0.68644; candidate accuracy: 0.74948; delta: **+0.06304**.
- Baseline macro-F1: 0.50691; candidate macro-F1: 0.53949; delta: **+0.03258**.
- Route rate: 0.29762; weighted helpful 279.8; harmful 86.4; net +193.4.

The optimum is not isolated. All first six configurations produce the same
predictions, while the neighboring `k=5` family retains +0.0602 accuracy and
+0.0298 macro-F1 at a lower 0.1875 route rate. All top ten use inverse
confidence. The first-ranked configuration is frozen as
`configs/v2/averitec_router_v1.json` before any holdout intervention is run.

This is still a train-only, population-reweighted estimate rather than a final
generalization result. The 216-configuration search can introduce selection
optimism, so bootstrap and negative-control checks are required next.
