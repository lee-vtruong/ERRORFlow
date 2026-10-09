# Step 02 — Prediction và execution-trace contracts

## Mục tiêu

Chuẩn hóa artifact đầu ra của verifier trước khi kết nối Qwen3. Mỗi prediction phải truy được claim, model/checkpoint, split, seed, evidence, token cost và latency. Mỗi trace phải ghi các bước workflow, số LLM call và cờ leakage.

## Đã làm

- `src/errorflow/contracts.py`: `PredictionRecord` và `ExecutionTrace`.
- `scripts/run_step2_demo.py`: tạo fixture deterministic, không phải kết quả thực nghiệm.
- `tests/test_contracts.py`: kiểm tra serialization và default leakage flags.

## Chạy trên server

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/run_step2_demo.py
find outputs/step2_demo -maxdepth 1 -type f -print
cat outputs/step2_demo/predictions.jsonl
```

Kỳ vọng: `6 passed` và 3 prediction records + 3 trace records.

## Lưu ý

Fixture chỉ kiểm tra format và pipeline. Không được dùng làm số liệu nghiên cứu hoặc baseline accuracy.
