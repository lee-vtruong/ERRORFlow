# Direct memory ablations for final R6-NLF

The historical B0--B17 studies used the original HardLabel compatibility
filter. The evaluators now expose that choice explicitly. Their default remains
`same_label` for exact reproduction; the commands below use
`--compatibility-mode none`, which is the final R6-NLF method.

Both runs use cached B1 and A3 predictions. They do not invoke Qwen again. Fold
exclusion, grouped duplicate isolation, prediction-level JSONL, helpful/harmful
counts, exact McNemar tests, paired bootstrap intervals, Holm correction, and
budget sweeps remain unchanged.

## 1. B0--B9 under NLF

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

mkdir -p outputs/v3/averitec/memory_ablation_nlf outputs/logs
set -o pipefail

python scripts/evaluate_v3_memory_ablation.py \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --output-dir outputs/v3/averitec/memory_ablation_nlf \
  --compatibility-mode none \
  --group-threshold 0.90 \
  --random-seeds $(seq 2026 2045) \
  --bootstrap-samples 5000 \
  --seed 2026 \
  2>&1 | tee outputs/logs/memory_ablation_nlf_b0_b9.log

echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

This run supplies B0 No Memory, B1 Random Memory, B7 Error-Only, and B9 Full
Outcome Memory needed by the final-method analysis. It also retains B2--B8 for
the appendix and artifact.

## 2. B10--B17 and budget sweep under NLF

```bash
mkdir -p outputs/v3/averitec/memory_controls_nlf
set -o pipefail

python scripts/evaluate_v3_memory_controls.py \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --output-dir outputs/v3/averitec/memory_controls_nlf \
  --compatibility-mode none \
  --group-threshold 0.90 \
  --random-seeds $(seq 2026 2045) \
  --budgets 0.05 0.10 0.1118 0.15 0.20 0.25 0.30 0.35 0.40 \
  --bootstrap-samples 5000 \
  --seed 2026 \
  2>&1 | tee outputs/logs/memory_ablation_nlf_b10_b17.log

echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

The confirmatory direct-final-method subset is B0, B1, B9, B10 Recovery-Only,
B14 Stratified Outcome Shuffle, B15 Class-Conditional Utility, and B16 Uniform
Outcome Weights. B7 remains a required budget-composition sensitivity. B11--B13
and B17 remain appendix controls.

## 3. Package the results for the paper update

```bash
python - <<'PY' | tee outputs/v3/averitec/NLF_MEMORY_RESULTS_FOR_CODEX.txt
import json
from pathlib import Path

for path in (
    Path("outputs/v3/averitec/memory_ablation_nlf/summary.json"),
    Path("outputs/v3/averitec/memory_controls_nlf/summary.json"),
):
    report = json.loads(path.read_text(encoding="utf-8"))
    print("\nSUMMARY", path)
    print("CONFIGURATION", json.dumps(report["configuration"], sort_keys=True))
    print("BASELINE", json.dumps(report["baseline"], sort_keys=True))
    results = report.get("primary_results", report.get("results", {}))
    for name, row in results.items():
        if "seed=" not in name or name.endswith("seed=2026"):
            print("RESULT", name, json.dumps(row, sort_keys=True))
    if "stochastic_aggregate" in report:
        print("STOCHASTIC", json.dumps(report["stochastic_aggregate"], sort_keys=True))
    if "primary_matched_comparisons_b9_minus_variant" in report:
        print("SIGNIFICANCE", json.dumps(
            report["primary_matched_comparisons_b9_minus_variant"], sort_keys=True
        ))
    if "budget_results" in report:
        print("BUDGET_RESULTS", json.dumps(report["budget_results"], sort_keys=True))
        print("BUDGET_SIGNIFICANCE", json.dumps(
            report["budget_comparisons_b9_minus_control"], sort_keys=True
        ))
PY

wc -l \
  outputs/v3/averitec/memory_ablation_nlf/predictions.jsonl \
  outputs/v3/averitec/memory_controls_nlf/predictions.jsonl \
  outputs/v3/averitec/memory_controls_nlf/budget_predictions.jsonl
```

Return `outputs/v3/averitec/NLF_MEMORY_RESULTS_FOR_CODEX.txt`. Results from
these commands replace HardLabel B-table claims only after the output confirms
`"compatibility_mode": "none"`.
