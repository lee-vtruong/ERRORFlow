# Step 09 — Prepare real FEVER subset with evidence text

## Mục tiêu

Tạo input JSONL thực từ FEVER claim file và Wikipedia corpus, không dùng fixture giả lập.

## Chạy trên server

```bash
cd ~/whale/ERRORFlow
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python scripts/prepare_fever_subset.py \
  --claims data/raw/fever/paper_dev.jsonl \
  --wiki-dir data/raw/fever/wiki-pages \
  --output data/processed/fever/paper_dev_first100.jsonl \
  --limit 100
```

Kiểm tra:

```bash
wc -l data/processed/fever/paper_dev_first100.jsonl
head -n 1 data/processed/fever/paper_dev_first100.jsonl
```

Mỗi record có `claim_id`, `claim`, `evidence`, `gold_label`, dataset/version và source split. Claim positive không resolve được evidence sẽ bị loại và được đếm trong `missing_positive_evidence`; NEI có thể không có evidence.

## Chưa chạy Qwen

Sau khi kiểm tra vài record đầu tiên, mới đưa file này vào baseline Qwen3. Đây là bước chuẩn bị dữ liệu, chưa phải kết quả đánh giá.
