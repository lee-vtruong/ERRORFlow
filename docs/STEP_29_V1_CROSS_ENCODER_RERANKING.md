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

## Kết quả reranking 1K

Sanity check đạt: sau rerank, any-sentence recall@100 vẫn là `91.04%` và complete-group recall@100 vẫn là `86.87%`.

| Pool/ranking | Any@1 | Any@5 | Complete@1 | Complete@5 |
|---|---:|---:|---:|---:|
| Union trước BGE | 35.52% | 62.39% | 33.13% | 58.51% |
| Union sau BGE | 42.39% | 68.06% | 38.51% | 63.58% |
| Delta | +6.87 | +5.67 | +5.37 | +5.07 |

BGE cải thiện đúng vùng ngân sách top-5 của verifier. Complete@10 cũng tăng từ `70.15%` lên `71.79%`; dao động nhỏ tại @20 không ảnh hưởng cấu hình đã khóa. Retrieval V1 được chốt cho baseline 1K là `title top-50 + lexical top-50 → deduplicated union-100 → BGE rerank → top-5`.
