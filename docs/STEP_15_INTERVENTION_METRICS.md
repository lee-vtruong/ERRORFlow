# Step 15 — Intervention recovery metrics

Chạy riêng cho từng intervention:

```bash
python scripts/evaluate_intervention.py \
  --errors outputs/qwen3_baseline/fever_train_first1000_retrieved_errors.jsonl \
  --intervention outputs/qwen3_interventions/evidence_critic.jsonl \
  --output outputs/qwen3_interventions/evidence_critic_metrics.json

python scripts/evaluate_intervention.py \
  --errors outputs/qwen3_baseline/fever_train_first1000_retrieved_errors.jsonl \
  --intervention outputs/qwen3_interventions/conflict_check.jsonl \
  --output outputs/qwen3_interventions/conflict_check_metrics.json
```

`recovery_rate` là số baseline errors được intervention sửa đúng chia cho tổng baseline errors. Đây là train-side intervention evidence, chưa được dùng để chọn policy trên final evaluation.
