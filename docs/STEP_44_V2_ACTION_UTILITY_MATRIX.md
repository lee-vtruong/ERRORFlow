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

Completed. The matrix shows strong action conditioning and rejects a global
prompt policy. No action recovered any error whose gold label was CONFLICTING,
while every action regressed 8/10 correct CONFLICTING controls. Coverage audit
was strongest for NEI errors predicted as REFUTED (9/10) or SUPPORTED (5/10).
Conflict audit was strongest for SUPPORTED→NEI (9/10), SUPPORTED→REFUTED (6/10),
and tied refutation audit on REFUTED→NEI (6/10). These gold transitions remain
offline annotations, not online router inputs.
