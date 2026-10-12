# Step 28 — V1.1 entity/page-title candidate expansion

## Kết luận từ lexical top-50 audit

Audit trên 1.000 `shared_task_dev` claims có 670 claim kiểm chứng được và 330 NEI. Lexical BM25 đạt:

| k | Any gold sentence recall | Complete evidence-group recall | Mean best-group coverage |
|---:|---:|---:|---:|
| 1 | 11.94% | 10.30% | 11.03% |
| 5 | 26.87% | 23.58% | 25.18% |
| 10 | 33.43% | 29.40% | 31.38% |
| 20 | 41.79% | 37.31% | 39.52% |
| 50 | 50.75% | 45.67% | 48.17% |

Reranker chỉ sắp xếp lại candidates đã có, nên không thể khắc phục khoảng 54% claim chưa có complete gold evidence ở top-50. Bước tiếp theo phải mở rộng candidate generation.

## Phương pháp V1.1

`retrieve_fever_titles.py` tạo các page-title candidates chỉ từ claim, quét FEVER wiki shards và lấy sentence từ các trang có tiêu đề khớp exact với claim n-gram. Không dùng gold label hoặc annotated evidence. Page có hậu tố disambiguation, ví dụ `Mercury_-LRB-planet-RRB-`, cũng expose base title `Mercury`.

`merge_retrieval_candidates.py` đặt tối đa 10 title candidates trước lexical BM25, loại duplicate theo `(page_id, sentence_id)`, rồi giữ top-50. Cả title-only và hybrid đều phải được audit để tách contribution.

## Lệnh chạy trên server

```bash
cd ~/whale/ERRORFlow
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python scripts/retrieve_fever_titles.py \
  --wiki-dir data/raw/fever/wiki-pages \
  --claims data/processed/fever/shared_task_dev_first1000_claims.jsonl \
  --output data/processed/fever/shared_task_dev_first1000_title_candidates50.jsonl \
  --k 50 \
  --max-ngram 6 \
  --progress-every 10

python scripts/evaluate_fever_retrieval.py \
  --gold data/raw/fever/shared_task_dev.jsonl \
  --retrieved data/processed/fever/shared_task_dev_first1000_title_candidates50.jsonl \
  --output outputs/retrieval_v1/shared_task_dev_first1000_title_metrics.json \
  --ks 1 5 10 20 50

python scripts/merge_retrieval_candidates.py \
  --lexical data/processed/fever/shared_task_dev_first1000_candidates50.jsonl \
  --title data/processed/fever/shared_task_dev_first1000_title_candidates50.jsonl \
  --output data/processed/fever/shared_task_dev_first1000_hybrid_candidates50.jsonl \
  --k 50 \
  --title-first 10

python scripts/evaluate_fever_retrieval.py \
  --gold data/raw/fever/shared_task_dev.jsonl \
  --retrieved data/processed/fever/shared_task_dev_first1000_hybrid_candidates50.jsonl \
  --output outputs/retrieval_v1/shared_task_dev_first1000_hybrid_metrics.json \
  --ks 1 5 10 20 50
```

## Cổng quyết định

- Hybrid complete-group recall@50 tăng rõ: giữ title expansion và tiếp tục cải thiện ranking/reranking.
- Title-only tốt nhưng hybrid top-k thấp: sweep `--title-first` trên artifact đã tạo, không cần quét wiki lại.
- Hybrid recall@50 vẫn thấp: bước sau là fuzzy/alias entity linking hoặc dense document retrieval, chưa chạy verifier full.
