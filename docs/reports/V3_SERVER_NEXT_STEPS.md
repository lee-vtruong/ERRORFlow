# ERRORFlow V3 server handoff — bounded Phase 1 pilot

> Current checkpoint (2026-10-13): the historical Phase 1 instructions below
> are complete. WP3 CLIMATE-FEVER is also complete: 1,535-example frozen
> four-label evaluation, full cached B1/A3 inference, source-only R6,
> matched-route controls, paired significance, and per-class diagnosis. The
> authoritative results are in `CLIMATE_FEVER_PROTOCOL.md` and
> `V3_CROSS_DOMAIN_VALIDATION.md`. Do not rerun or tune on CLIMATE-FEVER; it is
> now a consumed locked external benchmark. The next research operation must
> use a predeclared WP4 evidence-condition protocol or proceed to the paper
> package without target-specific tuning.
>
> WP4 is now complete. The next active deliverable is WP5 paper packaging;
> see `V3_PAPER_PACKAGE.md`. Any new model, retriever, prompt, or router is a
> separately versioned follow-up experiment and must not overwrite V3 results.

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
