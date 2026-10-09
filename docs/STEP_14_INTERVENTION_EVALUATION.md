# Step 14 — Intervention evaluation

## Mục tiêu

Đánh giá intervention trên đúng 376 lỗi, giữ nguyên Qwen3, evidence retrieval và claim. Chỉ instruction thay đổi.

## Chuẩn bị

Sau khi pull code mới, phải rebuild error list để records có evidence text:

```bash
python scripts/build_error_list.py \
  --gold data/processed/fever/train_first1000_retrieved.jsonl \
  --predictions outputs/qwen3_baseline/fever_train_first1000_retrieved.jsonl \
  --output outputs/qwen3_baseline/fever_train_first1000_retrieved_errors.jsonl
```

## Chạy intervention

```bash
python scripts/run_qwen3_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input outputs/qwen3_baseline/fever_train_first1000_retrieved_errors.jsonl \
  --output outputs/qwen3_interventions/evidence_critic.jsonl \
  --split train_error_recovery \
  --intervention evidence_critic

python scripts/run_qwen3_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input outputs/qwen3_baseline/fever_train_first1000_retrieved_errors.jsonl \
  --output outputs/qwen3_interventions/conflict_check.jsonl \
  --split train_error_recovery \
  --intervention conflict_check
```

Hai run này có thể mất khoảng 5–10 phút mỗi run. Kết quả không được chọn theo test; đây là train-side intervention evidence.
