# Step 37 — AVeriTeC four-label oracle-evidence smoke baseline

## Why this run is bounded

The first normalized file exposed duplicate IDs because official AVeriTeC rows
do not contain an ID field. Step 37 fixes IDs to stable `split:index` values and
adds hard empty/duplicate-ID checks before any model run.

This smoke baseline uses the dataset's annotated question-answer evidence. It
is therefore an **oracle-evidence verifier diagnostic**, not an end-to-end
retrieval score and not directly comparable to AVeriTeC shared-task systems.

## Server commands

After pulling the Step 37 commit:

```bash
cd ~/whale/ERRORFlow
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python -m pytest -q

python -m errorflow.cli normalize \
  --dataset averitec --split dev \
  --input data/raw/averitec/dev.json \
  --output data/processed/v2/averitec/dev_first25.jsonl \
  --limit 25

python - <<'PY'
import json
path = "data/processed/v2/averitec/dev_first25.jsonl"
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
ids = [row["example_id"] for row in rows]
print("rows:", len(rows))
print("unique_ids:", len(set(ids)))
print("first_id:", ids[0])
print("labels:", sorted(set(row["label"] for row in rows)))
PY
```

Required gate: `rows: 25`, `unique_ids: 25`, `first_id: dev:0`. Only after that:

```bash
python scripts/run_v2_qwen_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input data/processed/v2/averitec/dev_first25.jsonl \
  --output outputs/v2/averitec/oracle_dev_first25_predictions.jsonl \
  --label-set averitec \
  --split dev_first25

python scripts/evaluate_v2_predictions.py \
  --gold data/processed/v2/averitec/dev_first25.jsonl \
  --predictions outputs/v2/averitec/oracle_dev_first25_predictions.jsonl \
  --output outputs/v2/averitec/oracle_dev_first25_metrics.json
```

## Result

Pending server execution. Record the ID gate and metrics. Do not interpret 25
examples as a research result; it only validates four-label prompting, parsing,
I/O, and compatibility with the cached Qwen model.
