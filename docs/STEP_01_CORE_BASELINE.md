# Step 01 — Core ERRORFlow baseline

## Mục tiêu

Tạo lõi dependency-free để ERRORFlow có thể chạy và kiểm thử trước khi kết nối Qwen3, dataset hoặc retrieval.

## Đã làm

- `src/errorflow/schemas.py`: schema `ErrorMemoryEntry`.
- `src/errorflow/memory.py`: lưu/đọc error memory dạng JSONL.
- `src/errorflow/router.py`: router heuristic chọn intervention theo recovery rate thực nghiệm trừ cost.
- `src/errorflow/metrics.py`: baseline accuracy, final accuracy, Recovery Rate và Regression Rate.
- `tests/test_core.py`: 3 test cho persistence, routing và transition metrics.

## Nguyên tắc

- Router hiện tại minh bạch, không training và không gọi LLM.
- Router chỉ dùng memory được truyền vào; không tự đọc test labels.
- Cost hiện đo bằng `extra_tokens`; sau này mở rộng thêm số call, latency và USD.

## Cách chạy local/server

```bash
cd ~/whale/ERRORFlow
source ~/whale/GraphCURE/.venv/bin/activate
PYTHONPATH=src python -m pytest -q
```

Kỳ vọng: `4 passed` (gồm test schema hiện có và 3 test core mới).

## Chưa làm trong step này

- Chưa tải hoặc gọi Qwen3.
- Chưa đưa dataset thật vào memory.
- Chưa sinh diagnosis tự động.
- Chưa chạy evaluation trên validation/test.
