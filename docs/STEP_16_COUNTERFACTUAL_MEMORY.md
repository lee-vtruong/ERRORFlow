# Step 16 — Counterfactual intervention memory

## Build

```bash
python scripts/build_counterfactual_memory.py \
  --errors outputs/qwen3_baseline/fever_train_first1000_retrieved_errors.jsonl \
  --intervention evidence_critic outputs/qwen3_interventions/evidence_critic.jsonl \
  --intervention conflict_check outputs/qwen3_interventions/conflict_check.jsonl \
  --output outputs/error_memory/train_first1000_counterfactual.jsonl
```

Mỗi record giữ prediction của từng intervention, `recovery_success`, và `best_action`. Record được đánh dấu train observation; chưa dùng cho final evaluation.

## Kết quả hiện tại

- evidence_critic: 51/376 recovered (13.56%).
- conflict_check: 20/376 recovered (5.32%).

Không được kết luận evidence_critic luôn tốt hơn trên claim đúng ban đầu; cần đo regression trên toàn bộ train subset trước khi khóa router.
