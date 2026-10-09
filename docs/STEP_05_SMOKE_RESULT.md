# Step 05 — Qwen3 smoke result

## Kết quả

Qwen3-4B-Instruct-2507 đã load thành công trên server và xử lý claim mẫu:

```text
demo_001 → SUPPORTED
```

Output được ghi tại `outputs/qwen3_baseline/demo_predictions.jsonl` trên server.

## Quan sát kỹ thuật

- Model weights load thành công trên GPU.
- Generated output có lặp label, nhưng strict parser lấy đúng nhãn.
- `token_count=8`.
- Bản sửa kế tiếp đo latency thật và loại warning API deprecated.

## Diễn giải

Đây chỉ là smoke test để xác nhận model/pipeline hoạt động. Chưa được dùng làm accuracy hoặc kết luận nghiên cứu vì chưa có dataset gold và protocol evaluation.
