# P2 second-backbone validation protocol

## Scope and integrity rules

This is a separately versioned robustness experiment, not a replacement for
the frozen Qwen3 results. Select and record the second backbone before reading
any outcomes. The model must be a locally available causal instruction model
supported by Transformers and must expose a usable chat template.

The central constraint is that action outcomes are backbone-specific. Do not
reuse Qwen3 baseline scores, A3 predictions, or residual memory. Build B1, A3,
diagnoses, and memory anew for the second backbone. Freeze the same prompts,
group threshold 0.90, five folds, top-k 5, minimum support 2, utility threshold
0, and cost weight 0.1. No target-label tuning is allowed.

The minimum experiment is AVeriTeC grouped internal plus CLIMATE-FEVER official
candidates. CLIMATE-FEVER is preferred here because it uses the same four-label
space and avoids conflating backbone robustness with the SciFact adapter.

## 0. Declare the model and smoke-test it

Set `SECOND_MODEL` to an already downloaded local snapshot. Do not use a moving
Hub branch name in the final artifact.

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src
export SECOND_MODEL=/absolute/path/to/frozen/second-model-snapshot
export SECOND_TAG=second_backbone_v1

python scripts/run_v3_qwen_action.py \
  --model "$SECOND_MODEL" \
  --input data/processed/v2/averitec/train_all.jsonl \
  --action conflict_audit_v1 --split train \
  --output outputs/v3/backbones/$SECOND_TAG/smoke25/b1.jsonl \
  --max-samples 25 --max-llm-calls 25 --seed 2026 --dry-run
```

Then run the same command without `--dry-run`. Inspect the manifest, 25 output
rows, label-score fields, GPU memory, and latency before authorizing full runs.
Although the runner retains its historical `qwen` filename, its implementation
uses `AutoTokenizer` and `AutoModelForCausalLM`; successful smoke testing is the
compatibility gate.

## 1. Build backbone-specific AVeriTeC B1, A3, and memory

```bash
mkdir -p outputs/v3/backbones/$SECOND_TAG/averitec/{actions,development,eval} outputs/logs
set -o pipefail

python scripts/run_v3_qwen_action.py \
  --model "$SECOND_MODEL" \
  --input data/processed/v2/averitec/train_all.jsonl \
  --action conflict_audit_v1 --split train \
  --output outputs/v3/backbones/$SECOND_TAG/averitec/actions/b1.jsonl \
  --seed 2026 \
  2>&1 | tee outputs/logs/${SECOND_TAG}_averitec_b1.log

python scripts/run_v3_qwen_action.py \
  --model "$SECOND_MODEL" \
  --input data/processed/v2/averitec/train_all.jsonl \
  --action conflicting_rescue_v1 --split train \
  --output outputs/v3/backbones/$SECOND_TAG/averitec/actions/a3.jsonl \
  --seed 2026 \
  2>&1 | tee outputs/logs/${SECOND_TAG}_averitec_a3.log

python scripts/analyze_v3_residuals.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --predictions outputs/v3/backbones/$SECOND_TAG/averitec/actions/b1.jsonl \
  --output-dir outputs/v3/backbones/$SECOND_TAG/averitec/development \
  --folds 5 --seed 2026

python scripts/build_v3_residual_memory.py \
  --baseline-records outputs/v3/backbones/$SECOND_TAG/averitec/development/baseline_scores.jsonl \
  --candidate conflicting_rescue_v1=outputs/v3/backbones/$SECOND_TAG/averitec/actions/a3.jsonl \
  --output outputs/v3/backbones/$SECOND_TAG/averitec/actions/a3_memory.jsonl

python scripts/evaluate_v3_nlf_policy.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline outputs/v3/backbones/$SECOND_TAG/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/backbones/$SECOND_TAG/averitec/actions/a3.jsonl \
  --memory outputs/v3/backbones/$SECOND_TAG/averitec/actions/a3_memory.jsonl \
  --output-dir outputs/v3/backbones/$SECOND_TAG/averitec/eval \
  --protocol grouped_internal --group-threshold 0.90 \
  --samples 5000 --seed 2026
```

`evaluate_v3_nlf_policy.py` exports B1, A3-All, HardLabel, NLF, and the
matched confidence control from the same cached predictions.

## 2. Run one locked external condition

Generate both prompts with the same second backbone, but route using only the
backbone-specific AVeriTeC source memory.

```bash
mkdir -p outputs/v3/backbones/$SECOND_TAG/climate_fever/{actions,eval}

python scripts/run_v3_qwen_action.py \
  --model "$SECOND_MODEL" \
  --input data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --action conflict_audit_v1 --label-protocol averitec4 \
  --split external_climate_fever \
  --output outputs/v3/backbones/$SECOND_TAG/climate_fever/actions/b1.jsonl \
  --external-evaluation --seed 2026

python scripts/run_v3_qwen_action.py \
  --model "$SECOND_MODEL" \
  --input data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --action conflicting_rescue_v1 --label-protocol averitec4 \
  --split external_climate_fever \
  --output outputs/v3/backbones/$SECOND_TAG/climate_fever/actions/a3.jsonl \
  --external-evaluation --seed 2026

python scripts/evaluate_v3_nlf_policy.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --baseline outputs/v3/backbones/$SECOND_TAG/climate_fever/actions/b1.jsonl \
  --candidate outputs/v3/backbones/$SECOND_TAG/climate_fever/actions/a3.jsonl \
  --memory outputs/v3/backbones/$SECOND_TAG/averitec/actions/a3_memory.jsonl \
  --output-dir outputs/v3/backbones/$SECOND_TAG/climate_fever/eval \
  --protocol external --samples 5000 --seed 2026
```

## 3. Return checklist

```bash
wc -l \
  outputs/v3/backbones/$SECOND_TAG/averitec/actions/{b1,a3}.jsonl \
  outputs/v3/backbones/$SECOND_TAG/climate_fever/actions/{b1,a3}.jsonl \
  outputs/v3/backbones/$SECOND_TAG/averitec/actions/a3_memory.jsonl

cat outputs/v3/backbones/$SECOND_TAG/averitec/eval/summary.json
cat outputs/v3/backbones/$SECOND_TAG/climate_fever/eval/summary.json
df -h /
```

Report Accuracy, Macro-F1, route rate, helpful/harmful/net, per-class F1,
paired intervals, exact McNemar, average forward passes, and cached latency for
B1, A3-All, NLF, and Confidence-Only. If the model cannot score all frozen
labels reliably or lacks a compatible chat template, record that failure as a
backbone compatibility limitation rather than changing prompts after seeing
results.
