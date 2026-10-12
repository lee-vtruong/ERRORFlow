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

Completed on 155 train-only screening examples.

| Action | Helpful | Harmful | Net | Candidate accuracy | Candidate Macro-F1 | Latency |
|---|---:|---:|---:|---:|---:|---:|
| conflict audit | 34 | 12 | **+22** | **0.4000** | **0.3655** | 123.4 ms |
| coverage audit | 30 | 14 | +16 | 0.3613 | 0.3263 | 120.3 ms |
| refutation audit | 28 | 12 | +16 | 0.3613 | 0.3252 | 120.1 ms |

All actions have positive net change on the stratified screen. Conflict audit is
the aggregate leader, but it is not promoted to a global intervention because
the screen intentionally over-represents errors (baseline accuracy 0.2581).
Action-by-transition and control regression must be inspected before memory and
routing are built.
