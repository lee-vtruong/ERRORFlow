# Step 43 — Train-only intervention screen

## Baseline source

The frozen S0 train run paired all 3,068 examples and achieved accuracy 0.68644
and macro-F1 0.50691. It contains 962 errors. The largest transitions are
REFUTED→NEI (319), SUPPORTED→REFUTED (126), SUPPORTED→NEI (125),
REFUTED→SUPPORTED (105), and CONFLICTING→REFUTED (93).

## Screening design

Select up to 10 deterministic examples from every observed error transition and
10 baseline-correct controls from each gold label. This prevents the largest
class from hiding action-specific behavior and measures regression alongside
recovery. Selection uses seed 2026 and train only.

Three candidate actions are screened with the frozen S0 label scorer:

- `conflict_audit_v1`
- `coverage_audit_v1`
- `refutation_audit_v1`

No dev/holdout record is used for selection, prompting, or thresholds.

## Result

Pending server execution. An action is not accepted from recovery alone: net
correct change, control regression, per-transition utility, and four-forward-pass
cost are all required.
