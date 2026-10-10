# Step 21 — Shared-task dev holdout

## Mục tiêu

Đánh giá policy candidate học từ train trên FEVER `shared_task_dev`, không dùng annotated gold evidence để tạo input.

## Chuẩn bị 1,000 claim dev đầu tiên

```bash
python scripts/prepare_fever_claims.py \
  --input data/raw/fever/shared_task_dev.jsonl \
  --output data/processed/fever/shared_task_dev_first1000_claims.jsonl \
  --limit 1000
```

## Retrieval deterministic

```bash
python scripts/retrieve_fts.py \
  --db data/processed/fever/wiki_sentences.sqlite \
  --claims data/processed/fever/shared_task_dev_first1000_claims.jsonl \
  --output data/processed/fever/shared_task_dev_first1000_retrieved.jsonl \
  --k 5 --mode or --workers 4 --progress-every 25
```

Sau retrieval, chạy frozen Qwen3 baseline, tạo subset baseline-predicted `REFUTED`, áp dụng `evidence_critic`, merge và evaluate. Không thay đổi route dựa trên `paper_dev`.
