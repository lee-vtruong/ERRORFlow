# Step 30 — Run the V1 verifier baseline on 1K dev claims

## Mục tiêu

Đo `B1`: cùng Qwen3-4B verifier, prompt, decoding và 1.000 shared-task dev claims như V0/B0, chỉ thay evidence retrieval bằng pipeline V1 đã khóa:

```text
claim
  → exact page-title top-50 + lexical BM25 top-50
  → deduplicated union-100
  → BAAI/bge-reranker-v2-m3
  → top-5 evidence
  → frozen Qwen3-4B verifier
```

Không chạy evidence critic hoặc tái sử dụng router threshold V0 trong bước này.

## Tạo top-5 artifact

```bash
python scripts/select_retrieval_topk.py \
  --input data/processed/fever/shared_task_dev_first1000_union100_bge_reranked.jsonl \
  --output data/processed/fever/shared_task_dev_first1000_bge_top5.jsonl \
  --k 5

wc -l data/processed/fever/shared_task_dev_first1000_bge_top5.jsonl
```

Kỳ vọng đúng `1000` rows.

## Chạy frozen verifier

```bash
python scripts/run_qwen3_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input data/processed/fever/shared_task_dev_first1000_bge_top5.jsonl \
  --output outputs/qwen3_baseline/shared_task_dev_first1000_v1_bge_top5.jsonl \
  --split shared_task_dev_first1000_v1
```

Nếu bị ngắt, chạy lại cùng lệnh và thêm `--resume`.

## Đánh giá B1

```bash
python scripts/evaluate_predictions.py \
  --gold data/processed/fever/shared_task_dev_first1000_bge_top5.jsonl \
  --predictions outputs/qwen3_baseline/shared_task_dev_first1000_v1_bge_top5.jsonl \
  --output outputs/qwen3_baseline/shared_task_dev_first1000_v1_bge_top5_metrics.json

cat outputs/qwen3_baseline/shared_task_dev_first1000_v1_bge_top5_metrics.json
```

So sánh chính là B1 với B0 trên đúng 1.000 claims. Retrieval recall tăng không bảo đảm label accuracy tăng, vì verifier vẫn có thể suy luận sai hoặc bị distractor; vì vậy chưa xây ERRORFlow/E1 trước khi xem confusion matrix của B1.

## Kết quả B1

B1 đạt Accuracy `58.30%`, tăng `2.70` điểm so với B0, nhưng Macro-F1 giảm từ `54.07%` xuống `53.36%`. SUPPORTED và REFUTED F1 lần lượt tăng `5.17` và `3.49` điểm, trong khi NEI F1 giảm `10.78` điểm. Stronger retrieval cải thiện verifiable claims nhưng làm verifier overcommit khi gold là NEI. Audit tiếp theo được khóa tại `docs/STEP_31_V1_ERROR_AUDIT.md`.
