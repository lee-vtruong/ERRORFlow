# Step 40 — AVeriTeC oracle-evidence holdout baseline

## Split discipline

Examples `dev:0` through `dev:24` were used to select the prompt/decoding
protocol and are now frozen as a calibration slice. They must not be included in
the next reported estimate.

The evaluation slice is `dev:25` through `dev:499` (475 examples). This is a
project holdout carved from official dev, not the untouched official dev split.
The run still uses annotated QA evidence, so it measures verifier behavior under
oracle evidence rather than end-to-end retrieval.

## Protocol

- Model: Qwen3-4B-Instruct-2507, frozen.
- Prompt: plain four-label prompt (`instruction-profile none`).
- Decoding: S0 mean conditional label log-probability.
- Four model forward passes per example; zero generated tokens.

## Server commands

```bash
python -m errorflow.cli normalize \
  --dataset averitec --split dev \
  --input data/raw/averitec/dev.json \
  --output data/processed/v2/averitec/dev_holdout475.jsonl \
  --offset 25 --limit 475

python scripts/run_v2_qwen_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input data/processed/v2/averitec/dev_holdout475.jsonl \
  --output outputs/v2/averitec/oracle_dev_holdout475_s0.jsonl \
  --label-set averitec --split dev_holdout475 \
  --instruction-profile none --decoding label-score

python scripts/evaluate_v2_predictions.py \
  --gold data/processed/v2/averitec/dev_holdout475.jsonl \
  --predictions outputs/v2/averitec/oracle_dev_holdout475_s0.jsonl \
  --output outputs/v2/averitec/oracle_dev_holdout475_s0_metrics.json
```

## Required integrity checks

- Gold rows: 475; prediction rows: 475; unique IDs: 475.
- First/last IDs: `dev:25`, `dev:499`.
- Missing and extra predictions: zero.
- Average model forward passes: 4.0.

## Result

Pending server execution.
