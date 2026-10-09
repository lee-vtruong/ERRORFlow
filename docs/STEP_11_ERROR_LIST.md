# Step 11 — Build observable error list

## Mục tiêu

Chuyển các prediction sai thành records có thể audit, làm đầu vào cho error memory.

## Chạy trên server

```bash
cd ~/whale/ERRORFlow
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python scripts/build_error_list.py \
  --gold data/processed/fever/paper_dev_first100.jsonl \
  --predictions outputs/qwen3_baseline/fever_paper_dev_first100.jsonl \
  --output outputs/qwen3_baseline/fever_paper_dev_first100_errors.jsonl
```

## Nguyên tắc

`error_type_observable` chỉ mô tả nhãn/evidence quan sát được, ví dụ `NOT ENOUGH INFO → REFUTED`. Nó chưa phải diagnosis nguyên nhân. Không được coi raw reasoning của Qwen là ground truth về nguyên nhân lỗi.

Từ confusion matrix hiện tại, expected error list size là `36` records vì `99 - 63 = 36`.
