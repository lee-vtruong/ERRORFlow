# ERRORFlow V3 server handoff — bounded Phase 1 pilot

## Pull and verify

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src
python -m pytest -q
python -m errorflow.cli doctor
python -m errorflow.cli v3 audit --output outputs/v3/repository_audit.json
df -h /
```

Expected gate: **81 passed, 11 warnings**. Stop below 5 GB free space or on any
test/prompt-hash failure.

## Dry-run and execute a 25-example B1 pilot

```bash
python scripts/run_v3_qwen_action.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input data/processed/v2/averitec/train_all.jsonl \
  --action conflict_audit_v1 --split train \
  --output outputs/v3/averitec/smoke25/b1_predictions.jsonl \
  --max-samples 25 --max-llm-calls 25 --max-runtime-seconds 600 \
  --seed 2026 --dry-run

python scripts/run_v3_qwen_action.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input data/processed/v2/averitec/train_all.jsonl \
  --action conflict_audit_v1 --split train \
  --output outputs/v3/averitec/smoke25/b1_predictions.jsonl \
  --max-samples 25 --max-llm-calls 25 --max-runtime-seconds 600 \
  --seed 2026

python scripts/analyze_v3_residuals.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --predictions outputs/v3/averitec/smoke25/b1_predictions.jsonl \
  --output-dir outputs/v3/averitec/smoke25/development \
  --max-samples 25 --folds 5 --seed 2026
```

Return the action manifest and `residual_metrics.json` before launching full B1.

## Estimated next-stage cost

Historical AVeriTeC label scoring was roughly 0.11–0.12 seconds/example after
model load. A full B1 train cache is therefore approximately 6–10 minutes and
12,272 forward passes. A 155-example five-action screen is approximately 3,100
forward passes plus five model loads. Full five-action train inference would be
61,360 forward passes and is intentionally not authorized at this stage.
