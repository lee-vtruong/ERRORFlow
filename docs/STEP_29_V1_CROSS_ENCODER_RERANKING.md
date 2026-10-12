# Step 29 — V1 cross-encoder reranking

## Input và quyết định

Union pool trên 1.000 shared-task dev claims đạt complete evidence-group recall `86.87% @100` và any-sentence recall `91.04% @100`. So với title-only `82.69% @50`, lexical candidates bổ sung `4.18` điểm complete recall. Candidate pool này đủ mạnh để kiểm tra ranking.

Reranker được chọn cho lần thử đầu là `BAAI/bge-reranker-v2-m3`: off-the-shelf cross-encoder, không fine-tune trên FEVER. Model nhận cặp `(claim, evidence sentence)` và xuất relevance score; do đó hệ thống vẫn training-free đối với task FEVER.

Reranking giữ toàn bộ 100 candidates và provenance để audit recall@k. Chỉ sau khi đánh giá mới cắt top-5 làm đầu vào verifier.

## Tải model trên server

Model khoảng 2.29 GB. Kiểm tra dung lượng trước:

```bash
df -h /
mkdir -p /home/stackops/whale/cache/models/bge-reranker-v2-m3

hf download BAAI/bge-reranker-v2-m3 \
  --local-dir /home/stackops/whale/cache/models/bge-reranker-v2-m3

test -f /home/stackops/whale/cache/models/bge-reranker-v2-m3/config.json \
  && echo RERANKER_READY \
  || echo RERANKER_MISSING
```

## Rerank union-100

```bash
cd ~/whale/ERRORFlow
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python scripts/rerank_cross_encoder.py \
  --model /home/stackops/whale/cache/models/bge-reranker-v2-m3 \
  --input data/processed/fever/shared_task_dev_first1000_union_candidates100.jsonl \
  --output data/processed/fever/shared_task_dev_first1000_union100_bge_reranked.jsonl \
  --batch-size 32 \
  --max-length 512 \
  --device cuda \
  --local-files-only \
  --progress-every 10
```

Nếu job bị ngắt, chạy lại cùng lệnh và thêm `--resume`. Runner append từng claim, flush ngay và từ chối resume artifact corrupt/duplicate.

## Đánh giá ranking

```bash
python scripts/evaluate_fever_retrieval.py \
  --gold data/raw/fever/shared_task_dev.jsonl \
  --retrieved data/processed/fever/shared_task_dev_first1000_union100_bge_reranked.jsonl \
  --output outputs/retrieval_v1/shared_task_dev_first1000_bge_reranked_metrics.json \
  --ks 1 5 10 20 50 100
```

Sanity check bắt buộc: recall@100 sau rerank phải giống union trước rerank (`86.87%` complete và `91.04%` any-sentence), vì reranker chỉ thay đổi thứ tự. Nếu khác, artifact bị mất hoặc lệch candidates.

## Tạo verifier input sau khi ranking đạt yêu cầu

Chưa chạy bước này cho tới khi đã xem metrics reranking:

```bash
python scripts/select_retrieval_topk.py \
  --input data/processed/fever/shared_task_dev_first1000_union100_bge_reranked.jsonl \
  --output data/processed/fever/shared_task_dev_first1000_bge_top5.jsonl \
  --k 5
```

V1 baseline verifier phải dùng đúng top-5 này. Sau đó mới chạy Qwen, đánh giá B1 và xây lại error memory/router cho E1.
