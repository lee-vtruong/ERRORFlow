# Step 06 — FEVER dataset acquisition and audit

## Dataset source

FEVER 1.0 was downloaded from the official FEVER URLs. The raw files are not committed to Git and belong under `data/raw/fever/`.

## Downloaded files

- `train.jsonl`
- `shared_task_dev.jsonl`
- `paper_dev.jsonl`

## Audit command

```bash
cd ~/whale/ERRORFlow
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python scripts/audit_fever.py \
  data/raw/fever/train.jsonl \
  data/raw/fever/shared_task_dev.jsonl \
  data/raw/fever/paper_dev.jsonl \
  --summary outputs/fever/audit.json
```

## Important protocol note

FEVER records include claims, labels and evidence references. They do not automatically provide the evidence passage text in the downloaded claim JSONL. Do not pass evidence IDs as if they were evidence text. The next step must either connect the FEVER Wikipedia database/corpus or explicitly define a claim-only smoke protocol (which is not the main fact-verification experiment).
