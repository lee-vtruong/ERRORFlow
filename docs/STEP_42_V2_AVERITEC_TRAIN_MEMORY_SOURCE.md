# Step 42 — AVeriTeC train baseline for error-memory construction

## Objective

Run the frozen S0 verifier over all 3,068 official training examples. This is
not a reported generalization score; it creates the only label-aware source
permitted for intervention screening and error-memory construction.

## Server commands

```bash
python -m errorflow.cli normalize \
  --dataset averitec --split train \
  --input data/raw/averitec/train.json \
  --output data/processed/v2/averitec/train_all.jsonl

python scripts/run_v2_qwen_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input data/processed/v2/averitec/train_all.jsonl \
  --output outputs/v2/averitec/oracle_train_all_s0.jsonl \
  --label-set averitec --split train \
  --instruction-profile none --decoding label-score

python scripts/evaluate_v2_predictions.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --predictions outputs/v2/averitec/oracle_train_all_s0.jsonl \
  --output outputs/v2/averitec/oracle_train_all_s0_metrics.json
```

Use `--resume` if inference is interrupted. Expected runtime from the holdout
measurement is roughly 6–10 minutes, but progress output is authoritative.

## Guardrails

- IDs must run from `train:0` to `train:3067` and be unique.
- Do not include any dev record in the training memory.
- Do not run interventions on the sealed 475 holdout examples.
- Preserve all candidate intervention outcomes, including harmful ones.

## Result

Completed on all 3,068 official train examples with no missing or extra
predictions. S0 reached accuracy 0.68644 and macro-F1 0.50691. The weakest class
was CONFLICTING (F1 0.09322, recall 0.05641); REFUTED and SUPPORTED reached F1
0.78653 and 0.73241. This result is a train-only memory-construction baseline,
not a generalization claim.
