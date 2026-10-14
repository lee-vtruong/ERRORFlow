# Final R6-NLF ablations: N-C, N-D, and N-E

These runs use cached B1 and A3 predictions. They do not invoke Qwen. Every
memory lookup excludes the target claim and its grouped outer fold. The final
router uses no hard predicted-label compatibility filter.

## Common setup

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python -m pytest -q
mkdir -p outputs/v3/averitec/{nc_similarity,nd_utility,ne_sensitivity} outputs/logs
```

## N-C: final-NLF similarity components

NC0--NC8 use the same canonical `[\w'-]+` tokenizer and Group-C retrieval
engine. Natural mode uses each variant's own `U>0` decision. Matched mode uses
exactly the NC0 natural route count. McNemar, paired bootstrap, Holm correction,
and fold-level results are included in `summary.json`.

```bash
set -o pipefail
python scripts/evaluate_v3_substitution_ablation.py \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --examples data/processed/v2/averitec/train_all.jsonl \
  --output-dir outputs/v3/averitec/nc_similarity \
  --reference-variant NC0_full_nlf \
  --groups NC \
  --group-threshold 0.90 \
  --bootstrap-samples 5000 \
  --seed 2026 \
  2>&1 | tee outputs/logs/nc_similarity.log

echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

## N-D: utility and decision rule

ND3 uses source-only empirical prior shrinkage. ND5 is the predeclared utility
threshold grid. ND6 selects one threshold per target fold using nested source
folds only. `nd2_utility_diagnostics.jsonl` stores utility and rank changes
between smoothed and unsmoothed estimators.

```bash
set -o pipefail
python scripts/evaluate_v3_nlf_utility_ablation.py \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --examples data/processed/v2/averitec/train_all.jsonl \
  --output-dir outputs/v3/averitec/nd_utility \
  --threshold-grid -0.20 -0.10 -0.05 0.0 0.05 0.10 0.20 \
  --group-threshold 0.90 \
  --bootstrap-samples 5000 \
  --seed 2026 \
  2>&1 | tee outputs/logs/nd_utility.log

echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

Interpret ND1 as the same scientific question as NA6, but recomputed by the
canonical similarity engine so the utility-family comparison is self-contained.
Do not select a new threshold using external evaluation labels.

## N-E: sensitivity and robustness

NE4 memory size is the total number of source records sampled before target
fold exclusion. Every requested size is evaluated over 20 deterministic seeds.
The full-memory point is deterministic. NE6 is run separately on locked
external targets below; it is not estimated on the source population.

```bash
set -o pipefail
python scripts/evaluate_v3_nlf_sensitivity.py \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --examples data/processed/v2/averitec/train_all.jsonl \
  --output-dir outputs/v3/averitec/ne_sensitivity \
  --top-ks 1 3 5 10 20 \
  --min-supports 1 2 3 5 \
  --budgets 0.05 0.10 0.1222 0.15 0.20 0.30 0.40 \
  --memory-sizes 25 50 100 200 500 1000 \
  --random-seeds $(seq 2026 2045) \
  --group-thresholds 0.85 0.90 0.95 \
  --group-threshold 0.90 \
  --seed 2026 \
  2>&1 | tee outputs/logs/ne_sensitivity.log

echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

### NE6: cross-domain source-memory sensitivity

Run the source-memory subsets on the three locked external settings. The
script reports full memory, source predicted-label subsets, source gold-label
subsets, and topic subsets only when topic metadata actually exists. Target
labels are not read until after all routes have been materialized.

```bash
mkdir -p outputs/v3/averitec/ne6_external/{climate_fever,scifact_oracle,scifact_bm25}

python scripts/evaluate_v3_external_memory_sensitivity.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --baseline outputs/v3/climate_fever/full/conflict_audit_v1.jsonl \
  --candidate outputs/v3/climate_fever/full/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --output-dir outputs/v3/averitec/ne6_external/climate_fever \
  --samples 5000 --seed 2026 \
  2>&1 | tee outputs/logs/ne6_climate_fever.log

python scripts/evaluate_v3_external_memory_sensitivity.py \
  --gold data/processed/v3/scifact/dev_oracle.jsonl \
  --baseline outputs/v3/scifact/conflict_audit_v1.jsonl \
  --candidate outputs/v3/scifact/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --output-dir outputs/v3/averitec/ne6_external/scifact_oracle \
  --project-conflicting-to-nei \
  --samples 5000 --seed 2026 \
  2>&1 | tee outputs/logs/ne6_scifact_oracle.log

python scripts/evaluate_v3_external_memory_sensitivity.py \
  --gold data/processed/v3/scifact/dev_retrieved_bm25_top5.jsonl \
  --baseline outputs/v3/scifact/retrieved_evidence/conflict_audit_v1.jsonl \
  --candidate outputs/v3/scifact/retrieved_evidence/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --output-dir outputs/v3/averitec/ne6_external/scifact_bm25 \
  --project-conflicting-to-nei \
  --samples 5000 --seed 2026 \
  2>&1 | tee outputs/logs/ne6_scifact_bm25.log
```

This is the longest run because NE4 performs 120 seeded memory replays. It
prints progress for every configuration. It can be resumed only by rerunning;
no model inference is lost or repeated.

## Validate and export results

```bash
python - <<'PY' | tee outputs/v3/averitec/NC_ND_NE_RESULTS_FOR_CODEX.txt
import json
from pathlib import Path

paths = {
    "NC": Path("outputs/v3/averitec/nc_similarity/summary.json"),
    "ND": Path("outputs/v3/averitec/nd_utility/summary.json"),
    "NE": Path("outputs/v3/averitec/ne_sensitivity/summary.json"),
}
for group, path in paths.items():
    report = json.loads(path.read_text(encoding="utf-8"))
    print("\n===", group, path, "===")
    print("CONFIG", json.dumps(report["configuration"], sort_keys=True))
    print("BASELINE", json.dumps(report["baseline"], sort_keys=True))
    if group == "NC":
        for name, row in report["results"].items():
            print("RESULT", name, json.dumps(row, sort_keys=True))
        print("SIGNIFICANCE", json.dumps(
            report["matched_budget_reference_comparisons"], sort_keys=True
        ))
    elif group == "ND":
        for name, row in report["results"].items():
            print("RESULT", name, json.dumps(row, sort_keys=True))
        print("THRESHOLDS", json.dumps(report["threshold_sensitivity"], sort_keys=True))
        print("ND2", json.dumps(report["nd2_diagnostics"], sort_keys=True))
        print("SIGNIFICANCE", json.dumps(
            report["matched_budget_reference_comparisons"], sort_keys=True
        ))
    else:
        print("RESULTS", json.dumps(report["results"], sort_keys=True))
        print("MEMORY", json.dumps(report["memory_size_aggregate"], sort_keys=True))
        print("GROUPING", json.dumps(report["group_threshold_results"], sort_keys=True))
        print("UNCERTAINTY", json.dumps(report["utility_uncertainty"], sort_keys=True))
for path in sorted(Path("outputs/v3/averitec/ne6_external").glob("*/summary.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    print("\n=== NE6", path.parent.name, "===")
    print("RESULTS", json.dumps(report["results"], sort_keys=True))
    print("SIGNIFICANCE", json.dumps(report["comparisons"], sort_keys=True))
PY

wc -l \
  outputs/v3/averitec/nc_similarity/predictions.jsonl \
  outputs/v3/averitec/nd_utility/predictions.jsonl \
  outputs/v3/averitec/nd_utility/nd2_utility_diagnostics.jsonl \
  outputs/v3/averitec/ne_sensitivity/predictions.jsonl \
  outputs/v3/averitec/ne6_external/*/predictions.jsonl

df -h /
```

Send back `NC_ND_NE_RESULTS_FOR_CODEX.txt`. Keep all prediction-level JSONL
files for paper tables, paired checks, and figure generation.
