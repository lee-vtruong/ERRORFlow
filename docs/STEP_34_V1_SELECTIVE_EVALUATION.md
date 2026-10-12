# Step 34 — Merge and evaluate selective ERRORFlow V1

## Frozen policy

```text
if baseline prediction == REFUTED and confidence <= 0.90:
    run entailment_gate
else:
    keep B1 prediction
```

Expected route size is 49/1.000 claims (4.9%). Intervention predictions for all 382 REFUTED claims already exist; filter them to the frozen route rather than rerunning inference.

## Build the frozen route and filter predictions

```bash
python scripts/build_route_subset.py \
  --claims data/processed/fever/shared_task_dev_first1000_bge_top5.jsonl \
  --baseline outputs/qwen3_baseline/shared_task_dev_first1000_v1_bge_top5.jsonl \
  --prediction REFUTED \
  --max-confidence 0.90 \
  --output data/processed/fever/shared_task_dev_first1000_v1_refuted_c090_route.jsonl

python scripts/filter_predictions_by_claims.py \
  --claims data/processed/fever/shared_task_dev_first1000_v1_refuted_c090_route.jsonl \
  --predictions outputs/qwen3_interventions/v1_refuted_entailment_gate.jsonl \
  --output outputs/qwen3_interventions/v1_refuted_entailment_gate_c090.jsonl
```

Both commands must report 49 rows.

## Merge and evaluate E1

```bash
python scripts/merge_route_predictions.py \
  --baseline outputs/qwen3_baseline/shared_task_dev_first1000_v1_bge_top5.jsonl \
  --intervention outputs/qwen3_interventions/v1_refuted_entailment_gate_c090.jsonl \
  --output outputs/qwen3_interventions/shared_task_dev_first1000_v1_selective_full.jsonl

python scripts/evaluate_predictions.py \
  --gold data/processed/fever/shared_task_dev_first1000_bge_top5.jsonl \
  --predictions outputs/qwen3_interventions/shared_task_dev_first1000_v1_selective_full.jsonl \
  --output outputs/qwen3_interventions/shared_task_dev_first1000_v1_selective_metrics.json
```

Expected aggregate values from the sweep are Accuracy `58.30%` and Macro-F1 approximately `54.49%`; the full evaluator additionally reports per-class F1, cost and confusion.

## Paired bootstrap

```bash
python scripts/paired_bootstrap.py \
  --gold data/processed/fever/shared_task_dev_first1000_bge_top5.jsonl \
  --baseline outputs/qwen3_baseline/shared_task_dev_first1000_v1_bge_top5.jsonl \
  --candidate outputs/qwen3_interventions/shared_task_dev_first1000_v1_selective_full.jsonl \
  --output outputs/qwen3_interventions/shared_task_dev_first1000_v1_selective_bootstrap.json \
  --samples 5000 \
  --seed 2026
```

Report E1 only after verifying 1.000 unique predictions, route rate 4.9%, expected aggregate metrics and paired uncertainty.
