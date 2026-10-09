# Step 04 — Qwen3 baseline inference

## Mục tiêu

Chạy Qwen3 base model bằng `transformers`, không dùng LoRA adapter ở baseline đầu tiên. Mục đích là tạo frozen baseline sạch cho ERRORFlow.

## Đã làm

- `Qwen3TransformersVerifier` lazy-load model/tokenizer.
- Greedy decoding (`do_sample=False`).
- Ghi prediction với model, checkpoint, split và token count.
- Script `scripts/run_qwen3_baseline.py` đọc/ghi JSONL.

## Input format

```json
{"claim_id":"demo_001","claim":"...","evidence":["..."]}
```

## Chạy trên server

```bash
cd ~/whale/ERRORFlow
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

cat > /tmp/errorflow_demo.jsonl <<'EOF'
{"claim_id":"demo_001","claim":"The event happened in 2020.","evidence":["The event happened in 2020."]}
EOF

python scripts/run_qwen3_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input /tmp/errorflow_demo.jsonl \
  --output outputs/qwen3_baseline/demo_predictions.jsonl \
  --split smoke
```

## Lưu ý

- Chỉ chạy sau khi model đã tải xong tại `--model`.
- Nếu `config.json`/weights chưa có, dừng và gửi lỗi; không tự tải model trong script.
- Đây là smoke test, chưa phải benchmark.
