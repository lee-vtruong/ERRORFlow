# Step 10 — Evaluate Qwen3 baseline

## Chạy trên server

```bash
cd ~/whale/ERRORFlow
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python scripts/evaluate_predictions.py \
  --gold data/processed/fever/paper_dev_first100.jsonl \
  --predictions outputs/qwen3_baseline/fever_paper_dev_first100.jsonl \
  --output outputs/qwen3_baseline/fever_paper_dev_first100_metrics.json
```

## Báo cáo

Evaluator ghi Accuracy, Macro-F1, F1 từng lớp, confusion matrix, số prediction bị thiếu, average latency và average generated tokens. Chỉ các `claim_id` xuất hiện ở cả gold/prediction mới được tính paired metrics.

## Cảnh báo quan sát từ smoke run

Một số raw output dùng `NOT ENOUGH_INFO` trong reasoning dù label parser đã chuẩn hóa thành `NOT ENOUGH INFO`. Đây là dấu hiệu cần giữ label interface chặt hơn trong prompt/parser, nhưng không được sửa kết quả hậu nghiệm.
