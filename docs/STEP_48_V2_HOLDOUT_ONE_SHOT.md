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
