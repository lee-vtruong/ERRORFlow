# Step 03 — Verifier adapter contract

## Mục tiêu

Chuẩn hóa prompt, label parser và provenance record cho verifier Qwen3 trước khi tích hợp `transformers` model loading.

## Đã làm

- `src/errorflow/verifier.py`: `VerifierConfig`, prompt builder, parser cho ba nhãn và record builder.
- Không load model trong bước này; không tạo request/network call.
- Parser từ chối output không chứa label hợp lệ thay vì tự đoán.
- Record giữ model, checkpoint, seed, split, token count, latency và raw output.

## Kiểm tra trên server

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
PYTHONPATH=src python -m pytest -q
```

Kỳ vọng: `9 passed`.

## Bước kế tiếp

Tạo `Qwen3TransformersVerifier` có load local model path, greedy decoding, batch inference và ghi `predictions.jsonl`/`traces.jsonl`. Chỉ chạy sau khi xác nhận base model đã tải xong.
