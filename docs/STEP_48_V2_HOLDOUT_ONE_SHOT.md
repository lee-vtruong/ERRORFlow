# Step 48 — One-shot sealed holdout evaluation

## Frozen comparison

Evaluate exactly three systems on the 475-example AVeriTeC holdout:

1. Frozen S0 baseline.
2. Frozen ERRORFlow router using train-only memory.
3. `always_conflict_audit_v1` as the strongest global-action control.

The router configuration, prompts, memory, thresholds, and comparison list were
fixed before candidate holdout outcomes were generated. After this run, do not
tune V1 against the holdout.

## Generate candidate actions

Run the following command three times, changing the profile and output name:

```bash
python scripts/run_v2_qwen_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input data/processed/v2/averitec/dev_holdout475.jsonl \
  --output outputs/v2/averitec/holdout475_conflict_audit_v1.jsonl \
  --label-set averitec --split dev_holdout475 \
  --instruction-profile conflict_audit_v1 --decoding label-score
```

Repeat with `coverage_audit_v1` and `refutation_audit_v1`. Use `--resume` only
when resuming the same profile/output pair.

### Candidate-generation result

Completed on the server before router replay and final metric inspection. Each
frozen action produced exactly 475 predictions; total candidate rows were 1,425.
No prompt, memory record, threshold, or router setting was changed afterward.

## Replay frozen router

```bash
python scripts/run_v2_frozen_router.py \
  --input data/processed/v2/averitec/dev_holdout475.jsonl \
  --baseline outputs/v2/averitec/oracle_dev_holdout475_s0.jsonl \
  --memory outputs/v2/averitec/memory/action_outcomes_train.jsonl \
  --candidate \
    conflict_audit_v1=outputs/v2/averitec/holdout475_conflict_audit_v1.jsonl \
    coverage_audit_v1=outputs/v2/averitec/holdout475_coverage_audit_v1.jsonl \
    refutation_audit_v1=outputs/v2/averitec/holdout475_refutation_audit_v1.jsonl \
  --config configs/v2/averitec_router_v1.json \
  --output outputs/v2/averitec/holdout475_errorflow_router_v1.jsonl
```

## Evaluate without tuning

Use `evaluate_v2_predictions.py` for baseline, ERRORFlow, and the conflict
candidate. Then run `paired_bootstrap_v2.py` for baseline→ERRORFlow and
baseline→always-conflict. Also compare always-conflict→ERRORFlow to quantify the
quality/cost trade-off. Use 5,000 samples and seed 2026 for every comparison.

Acceptance is descriptive, not a new tuning gate: report accuracy, macro-F1,
per-class scores, helpful/harmful counts, route rate, average forward passes,
latency, paired confidence intervals, and cost-adjusted utility at weight 0.10.
For always-conflict, report both interpretations: eight forward passes as a
baseline-plus-audit cascade, and four passes when deployed as a standalone
replacement verifier. The latter is the stricter practical comparator.

## Final result

The one-shot evaluation paired all 475 examples with no missing or extra IDs.

| System | Accuracy | Macro-F1 | Avg forward passes | Avg latency |
|---|---:|---:|---:|---:|
| Frozen S0 baseline | 0.68632 | 0.52417 | 4.000 | 114.69 ms |
| ERRORFlow router V1 | 0.72421 | 0.55839 | 5.406 | 158.85 ms |
| Conflict audit standalone | **0.75789** | **0.56392** | **4.000** | **120.12 ms** |

Baseline to ERRORFlow produced 38 helpful and 20 harmful changes, net +18.
Accuracy improved by +0.03789 with CI95% `[0.00842, 0.06947]` and
`P(delta > 0)=0.992`. Macro-F1 improved by +0.03422, but its CI95%
`[-0.00288, 0.07245]` includes zero.

Baseline to conflict audit produced 56 helpful and 22 harmful changes, net +34.
Accuracy improved by +0.07158 with CI95% `[0.03789, 0.10737]`. Its macro-F1
gain of +0.03975 was not conclusive because CI95% was
`[-0.00941, 0.08730]`.

ERRORFlow was 0.03368 accuracy below standalone conflict audit; the paired
CI95% for ERRORFlow minus conflict was `[-0.06316, -0.00632]`, so this gap is
statistically meaningful. Their macro-F1 difference was only -0.00553 and was
not significant.

ERRORFlow routed 167/475 examples (35.16%), inferred from its 5.406 average
forward passes. It improved CONFLICTING F1 to 0.23256 and retained a more
balanced profile than conflict audit, whose CONFLICTING F1 was 0.10. However,
the standalone conflict verifier is the practical winner: higher accuracy,
similar macro-F1, lower latency, and fewer forward passes.

At the frozen cost weight 0.10 under cascade accounting, ERRORFlow's
accuracy utility is approximately 0.68905 versus baseline 0.68632, but its
macro-F1 utility is approximately 0.52323 versus baseline 0.52417. This narrow
cost-adjusted result does not overturn the standalone conflict comparison.

V1 is now closed. The holdout remains a reported evaluation artifact and must
not be used to tune a revised router.
