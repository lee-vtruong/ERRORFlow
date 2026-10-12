# Step 36 — AVeriTeC acquisition and 25-example smoke set

## Objective

Acquire only the official train/dev annotation files, record byte-level provenance, and normalize a bounded 25-example development slice. Do not download the multi-gigabyte knowledge store yet.

## Server commands

```bash
cd ~/whale/ERRORFlow
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python scripts/download_averitec.py --output-dir data/raw/averitec --splits train dev | tee outputs/logs/averitec_download.log

python -m errorflow.cli normalize --dataset averitec --split dev \
  --input data/raw/averitec/dev.json \
  --output data/processed/v2/averitec/dev_first25.jsonl --limit 25

wc -l data/processed/v2/averitec/dev_first25.jsonl
head -n 1 data/processed/v2/averitec/dev_first25.jsonl
cat data/raw/averitec/manifest.json
df -h /
```

## Expected

- `AVERITEC_READY` is printed and the manifest records URL, rows, labels, bytes, and SHA-256.
- The normalized smoke file has exactly 25 rows.
- No knowledge-store archive or new model is downloaded.

## Result

Pending server execution. Paste the manifest, row count, first normalized record, and disk report before baseline inference.
