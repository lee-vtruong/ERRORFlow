# Step 17 — Recovery/regression control evaluation

## Tách control set

```bash
python scripts/build_control_subset.py \
  --claims data/processed/fever/train_first1000_retrieved.jsonl \
  --baseline outputs/qwen3_baseline/fever_train_first1000_retrieved.jsonl \
  --output data/processed/fever/train_baseline_correct.jsonl
```

Expected: `control_records=619`.

## Chạy interventions trên control

Dùng `run_qwen3_baseline.py` với `--intervention evidence_critic` và `conflict_check`, output lần lượt vào `outputs/qwen3_interventions/*_control.jsonl`.

## Đánh giá đầy đủ

```bash
python scripts/evaluate_regression.py \
  --gold data/processed/fever/train_first1000_retrieved.jsonl \
  --baseline outputs/qwen3_baseline/fever_train_first1000_retrieved.jsonl \
  --intervention outputs/qwen3_interventions/evidence_critic_control.jsonl \
  --output outputs/qwen3_interventions/evidence_critic_full_metrics.json
```

`net_correct_change = recovered - regressed` là số đo đơn giản để sàng lọc intervention; policy cuối còn phải cân nhắc token cost và latency.
