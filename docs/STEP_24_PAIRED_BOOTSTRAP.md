# Step 24 — Paired bootstrap comparison

Compare dev baseline and frozen cost-aware route with paired bootstrap. Report point estimate, 95% interval, directional probability, and helpful/harmful counts.

```bash
python scripts/paired_bootstrap.py \
  --gold data/processed/fever/shared_task_dev_first1000_retrieved.jsonl \
  --baseline outputs/qwen3_baseline/shared_task_dev_first1000_confidence.jsonl \
  --candidate outputs/qwen3_interventions/shared_task_dev_cost_full.jsonl \
  --output outputs/qwen3_interventions/shared_task_dev_cost_bootstrap.json \
  --samples 5000 --seed 2026
```

The 1,000-claim result is a development audit, not final evidence. Do not retune the frozen threshold based on bootstrap output.
