# Step 27 — V1 retrieval audit before reranking

## Mục tiêu

Không tải hoặc gắn reranker một cách suy đoán. Trước hết đo xem gold evidence có xuất hiện trong danh sách ứng viên lexical hay không. Retriever nay lưu `page_id`, `sentence_id`, rank và BM25 score bên cạnh evidence text; contract đầu vào của verifier vẫn giữ nguyên.

Ba số được đo tại `k = 1, 5, 10, 20, 50` trên các claim SUPPORTS/REFUTES:

- `claim_any_sentence_recall`: top-k chứa ít nhất một gold sentence;
- `complete_evidence_group_recall`: top-k chứa trọn ít nhất một evidence group hợp lệ của FEVER;
- `mean_best_group_coverage`: coverage trung bình của evidence group được tìm thấy tốt nhất.

NEI không có annotated positive evidence nên được đếm riêng và không đưa vào mẫu số retrieval recall.

## Chạy audit 1K trên server

Sau khi pull code mới:

```bash
cd ~/whale/ERRORFlow
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python scripts/retrieve_fts.py \
  --db data/processed/fever/wiki_sentences.sqlite \
  --claims data/processed/fever/shared_task_dev_first1000_claims.jsonl \
  --output data/processed/fever/shared_task_dev_first1000_candidates50.jsonl \
  --k 50 \
  --mode or \
  --max-query-tokens 8 \
  --workers 4 \
  --progress-every 25

python scripts/evaluate_fever_retrieval.py \
  --gold data/raw/fever/shared_task_dev.jsonl \
  --retrieved data/processed/fever/shared_task_dev_first1000_candidates50.jsonl \
  --output outputs/retrieval_v1/shared_task_dev_first1000_candidates50_metrics.json \
  --ks 1 5 10 20 50
```

Không ghi đè artifact top-5 V0. File candidates50 là artifact audit V1 riêng.

## Cổng quyết định V1

- Nếu complete-group recall tiếp tục tăng mạnh từ `@5` đến `@50`, candidate generator đã tìm thấy nhiều gold evidence nhưng xếp hạng kém: thêm off-the-shelf reranker là bước hợp lý.
- Nếu cả any-sentence và complete-group recall `@50` vẫn thấp, reranker không thể cứu evidence chưa được retrieve: cần sửa entity/document retrieval hoặc query generation trước.
- Chỉ sau khi chọn cấu hình retrieval bằng development audit mới chạy lại frozen verifier và xây lại ERRORFlow artifacts. Không tái sử dụng router threshold V0 như kết luận mặc định cho V1.
