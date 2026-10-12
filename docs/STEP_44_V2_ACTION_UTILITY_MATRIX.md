# Step 44 — Action utility matrix

## Objective

Turn the three aggregate screening results into a per-group action matrix. Error
groups measure recovery; correct-control groups measure regression. The matrix
is an offline diagnostic only: gold transition names are not available to the
online router and cannot become route features.

## Command

```bash
python scripts/compare_v2_interventions.py \
  --metrics \
    conflict_audit_v1=outputs/v2/averitec/intervention_screen/conflict_audit_v1_metrics.json \
    coverage_audit_v1=outputs/v2/averitec/intervention_screen/coverage_audit_v1_metrics.json \
    refutation_audit_v1=outputs/v2/averitec/intervention_screen/refutation_audit_v1_metrics.json \
  --output outputs/v2/averitec/intervention_screen/action_utility_matrix.json \
  | tee outputs/logs/averitec_action_utility_matrix.log
```

## Leakage boundary

Gold labels may evaluate and annotate train outcomes, but an online decision may
only use observable fields such as claim/evidence representation, baseline
prediction, label-score margin, retrieved train memories, and expected cost.
Routing directly on `gold→prediction` would be invalid.

## Result

Pending server execution.
