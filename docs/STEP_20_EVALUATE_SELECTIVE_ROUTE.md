# Step 20 — Evaluate selective REFUTED route

Run evidence critic only on `train_predicted_refuted.jsonl`, then merge those predictions back into the full baseline before evaluation.

```bash
python scripts/run_qwen3_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input data/processed/fever/train_predicted_refuted.jsonl \
  --output outputs/qwen3_interventions/evidence_critic_refuted_route.jsonl \
  --split train_selective_route \
  --intervention evidence_critic

python scripts/merge_route_predictions.py \
  --baseline outputs/qwen3_baseline/fever_train_first1000_retrieved.jsonl \
  --intervention outputs/qwen3_interventions/evidence_critic_refuted_route.jsonl \
  --output outputs/qwen3_interventions/evidence_critic_selective_full.jsonl

python scripts/evaluate_predictions.py \
  --gold data/processed/fever/train_first1000_retrieved.jsonl \
  --predictions outputs/qwen3_interventions/evidence_critic_selective_full.jsonl \
  --output outputs/qwen3_interventions/evidence_critic_selective_metrics.json
```

Merged routed records must account for both baseline and intervention calls. The merge script sums token count and latency and records `llm_calls=2`; non-routed records use one call.
