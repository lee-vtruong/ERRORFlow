# Step 18 — Selective action analysis

## Chạy trên server

```bash
python scripts/analyze_action_by_error.py \
  --errors outputs/qwen3_baseline/fever_train_first1000_retrieved_errors.jsonl \
  --intervention outputs/qwen3_interventions/evidence_critic.jsonl \
  --output outputs/qwen3_interventions/evidence_critic_by_error_type.json
```

Mỗi nhóm báo cáo `n`, `recovered`, `changed` và `recovery_rate`. Nhóm nhỏ không đủ tin cậy; dùng kết quả để hình thành hypothesis/router feature, không tuning final test.
