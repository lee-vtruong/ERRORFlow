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

## Kết quả 1K

Title retrieval hoàn tất `1.000` rows, có candidates cho `987` claims và match `11.561` wiki pages. Trên 670 verifiable claims:

| Retrieval | Complete@1 | Complete@5 | Complete@10 | Complete@20 | Complete@50 |
|---|---:|---:|---:|---:|---:|
| Lexical BM25 | 10.30% | 23.58% | 29.40% | 37.31% | 45.67% |
| Title-only | 32.84% | 57.76% | 69.25% | 77.61% | 82.69% |
| Hybrid, title-first=10 | 33.13% | 58.51% | 70.15% | 72.84% | 74.33% |

Title-only tăng complete-group recall@50 thêm `37.01` điểm so với lexical. Hybrid tốt hơn nhẹ tại top-1/5/10, nhưng thấp hơn title-only tại top-20/50 do merge order: 10 title candidates đầu rồi lexical chiếm các vị trí còn lại, khiến title ranks 11–50 bị truncate. Đây là hiệu ứng composition, không phải bằng chứng lexical candidates làm giảm chất lượng gold pool.

Quyết định tiếp theo là tạo union pool tối đa 100 candidates bằng `title-first=50`, rồi đo recall@100. Pool này giữ nguyên toàn bộ title top-50 và bổ sung các lexical candidates không trùng; nó là đầu vào hợp lý cho reranker.

Union-100 thực tế đạt any-sentence recall `91.04%`, complete evidence-group recall `86.87%` và mean best-group coverage `88.94%`. So với title-only top-50, lexical candidates bổ sung `4.18` điểm complete recall. Candidate generation được chấp nhận cho V1 reranker experiment; bước tiếp theo được mô tả trong `docs/STEP_29_V1_CROSS_ENCODER_RERANKING.md`.
