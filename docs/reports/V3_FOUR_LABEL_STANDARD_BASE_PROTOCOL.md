# Four-label standard-Base prompt control

This is a predeclared diagnostic for the largest remaining prompt-confound
risk. It does not modify the frozen ERRORFlow method or any existing cache.
The control asks whether a conventional four-label verdict prompt, evaluated
with the same model, evidence, label scoring, and four forward passes, already
matches or exceeds the audit-style Base and the complete Memory-Global system.

## Decision rule

Accuracy is the primary endpoint and macro-F1 is secondary. Interpret each
dataset separately.

- **Pipeline dominated:** Standard Base exceeds Memory-Global and the paired
  95% accuracy interval is strictly positive. Remove that dataset's router
  quality-gain claim from the main paper; retain it only as prompt sensitivity.
- **Prompt confound unresolved:** Standard Base has a higher point estimate
  but the interval crosses zero. Report the control and avoid claiming that
  routing is better than a conventional Base.
- **Main result robust to prompt wording:** Memory-Global exceeds Standard
  Base with a strictly positive paired interval, or Standard Base does not
  improve Audit Base and remains below Memory-Global.

Do not tune or rewrite the standard prompt after seeing either dataset.

## Setup

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MODEL=/home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507
export OUT=outputs/v3/reviewer_round5/four_label_standard
mkdir -p "$OUT" outputs/logs
```

## Materialize the frozen standard Base

```bash
python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v2/averitec/train_all.jsonl \
  --output "$OUT/averitec.jsonl" \
  --action standard_verdict_v1 \
  --label-protocol averitec4 \
  --split train --seed 2026 \
  2>&1 | tee outputs/logs/reviewer5_standard_averitec.log
echo "AV_PIPELINE_EXIT=${PIPESTATUS[0]}"

python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --output "$OUT/climate_fever.jsonl" \
  --action standard_verdict_v1 \
  --label-protocol averitec4 \
  --split external_climate_fever_standard --external-evaluation --seed 2026 \
  2>&1 | tee outputs/logs/reviewer5_standard_climate.log
echo "CF_PIPELINE_EXIT=${PIPESTATUS[0]}"

wc -l "$OUT/averitec.jsonl" "$OUT/climate_fever.jsonl"

python - <<'PY' | tee "$OUT/prediction_distributions.txt"
import json, os
from collections import Counter

root = os.environ["OUT"]
for name in ("averitec", "climate_fever"):
    rows = [json.loads(line) for line in open(f"{root}/{name}.jsonl", encoding="utf-8") if line.strip()]
    print(name, "rows=", len(rows), "predictions=", dict(Counter(row["prediction"] for row in rows)))
PY
```

Expected row counts are 3,068 and 1,535.

## AVeriTeC paired and grouped comparisons

```bash
python scripts/evaluate_v3_grouped_pair.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate "$OUT/averitec.jsonl" \
  --baseline-name Audit-Base --candidate-name Standard-Base \
  --group-threshold 0.90 --samples 5000 --seed 2026 \
  --output "$OUT/averitec_standard_vs_audit.json"

python scripts/evaluate_v3_grouped_pair.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline "$OUT/averitec.jsonl" \
  --candidate outputs/v3/meta_routing_experiments/qwen3_4b/averitec_grouped090/per_query_predictions.jsonl \
  --candidate-filter policy=D1_NLF --candidate-filter mode=natural \
  --baseline-name Standard-Base --candidate-name Memory-Global \
  --group-threshold 0.90 --samples 5000 --seed 2026 \
  --output "$OUT/averitec_memory_vs_standard.json"

python scripts/evaluate_v3_grouped_pair.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline "$OUT/averitec.jsonl" \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --baseline-name Standard-Base --candidate-name Alt-All \
  --group-threshold 0.90 --samples 5000 --seed 2026 \
  --output "$OUT/averitec_alt_vs_standard.json"
```

## CLIMATE-FEVER paired and grouped comparisons

```bash
python scripts/evaluate_v3_grouped_pair.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --baseline outputs/v3/climate_fever/full/conflict_audit_v1.jsonl \
  --candidate "$OUT/climate_fever.jsonl" \
  --baseline-name Audit-Base --candidate-name Standard-Base \
  --group-threshold 0.90 --samples 5000 --seed 2026 \
  --output "$OUT/climate_standard_vs_audit.json"

python scripts/evaluate_v3_grouped_pair.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --baseline "$OUT/climate_fever.jsonl" \
  --candidate outputs/v3/meta_routing_experiments/qwen3_4b/climate_fever/per_query_predictions.jsonl \
  --candidate-filter policy=D1_NLF --candidate-filter mode=natural \
  --baseline-name Standard-Base --candidate-name Memory-Global \
  --group-threshold 0.90 --samples 5000 --seed 2026 \
  --output "$OUT/climate_memory_vs_standard.json"

python scripts/evaluate_v3_grouped_pair.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --baseline "$OUT/climate_fever.jsonl" \
  --candidate outputs/v3/climate_fever/full/conflicting_rescue_v1.jsonl \
  --baseline-name Standard-Base --candidate-name Alt-All \
  --group-threshold 0.90 --samples 5000 --seed 2026 \
  --output "$OUT/climate_alt_vs_standard.json"
```

## Export for paper update

```bash
{
  cat "$OUT/averitec.manifest.json"
  cat "$OUT/climate_fever.manifest.json"
  cat "$OUT/prediction_distributions.txt"
  for path in "$OUT"/*_vs_*.json; do
    echo "=== $path ==="
    cat "$path"
  done
} | tee "$OUT/RESULTS_FOR_CODEX.txt"
```

Return `RESULTS_FOR_CODEX.txt`, both manifests, and the two prediction label
distributions. No existing main-paper result should be removed or promoted
until these frozen comparisons have been inspected.
