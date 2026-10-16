# V3 reviewer round-2 controls

This protocol addresses three unresolved comparisons without changing Base,
Alt, memory, cached predictions, or frozen manifests:

1. outcome-balanced and Base-label-calibrated logistic routing;
2. protocol-separated quality--compute frontiers; and
3. group-safe per-backbone selection of same-label versus global-neighbor
   compatibility.

All learned/calibrated quantities use source folds only. External target labels
are read only after routes and predictions are materialized.

## 0. Setup

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
set -o pipefail

python -m pytest -q
python scripts/evaluate_v3_supervised_controls.py --help
python scripts/evaluate_v3_quality_budget_controls.py --help
python scripts/select_v3_compatibility_rule.py --help
mkdir -p outputs/v3/reviewer_round2 outputs/v3/figures/reviewer_round2 outputs/logs
```

## 1. Balanced logistic controls

The evaluator now emits five supervised controls. The two new controls are:

- `Logistic-BalancedOutcome`: inverse-frequency weighting over harmful,
  neutral, and helpful source outcomes, fitted independently inside each
  allowed source fold;
- `Logistic-PerLabelRank`: the plain logistic score converted to its empirical
  percentile within the Base-predicted label, using source scores only. This is
  a deployable label-conditional threshold/ranking control.

### 1.1 Grouped AVeriTeC

```bash
python scripts/evaluate_v3_supervised_controls.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/averitec_grouped090 \
  --protocol grouped_internal --route-count 375 \
  --bootstrap-samples 5000 --seed 2026 \
  --output-dir outputs/v3/reviewer_round2/averitec/supervised_375 \
  2>&1 | tee outputs/logs/reviewer2_supervised_averitec.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

### 1.2 Frozen external evaluations

```bash
COMMON_SOURCE=(
  --source-gold data/processed/v2/averitec/train_all.jsonl
  --source-baseline outputs/v3/averitec/development/baseline_scores.jsonl
  --source-candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl
  --source-meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/averitec_grouped090
  --protocol external --bootstrap-samples 5000 --seed 2026
)

python scripts/evaluate_v3_supervised_controls.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --baseline outputs/v3/climate_fever/full/conflict_audit_v1.jsonl \
  --candidate outputs/v3/climate_fever/full/conflicting_rescue_v1.jsonl \
  --meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/climate_fever \
  "${COMMON_SOURCE[@]}" \
  --output-dir outputs/v3/reviewer_round2/climate_fever/supervised

python scripts/evaluate_v3_supervised_controls.py \
  --gold data/processed/v3/scifact/dev_oracle.jsonl \
  --baseline outputs/v3/scifact/conflict_audit_v1.jsonl \
  --candidate outputs/v3/scifact/conflicting_rescue_v1.jsonl \
  --meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/scifact_oracle \
  "${COMMON_SOURCE[@]}" --project-conflicting-to-nei \
  --output-dir outputs/v3/reviewer_round2/scifact_oracle/supervised

python scripts/evaluate_v3_supervised_controls.py \
  --gold data/processed/v3/scifact/dev_retrieved_bm25_top5.jsonl \
  --baseline outputs/v3/scifact/retrieved_evidence/conflict_audit_v1.jsonl \
  --candidate outputs/v3/scifact/retrieved_evidence/conflicting_rescue_v1.jsonl \
  --meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/scifact_bm25 \
  "${COMMON_SOURCE[@]}" --project-conflicting-to-nei \
  --output-dir outputs/v3/reviewer_round2/scifact_bm25/supervised
```

Each `summary.json` contains helpful/harmful/net counts, per-class F1, exact
McNemar tests, ordinary paired 95% bootstrap intervals, Holm-adjusted accuracy
p-values, and conservative Bonferroni familywise bootstrap intervals for the
complete supervised-control family.

## 2. Quality--compute frontiers

The frontier evaluator does not create score-ensemble predictions. Build the
three external caches first, using the AVeriTeC-selected mixture coefficient
without reading target labels. These commands make no LLM calls.

```bash
SOURCE_ENSEMBLE=(
  --source-gold data/processed/v2/averitec/train_all.jsonl
  --source-baseline outputs/v3/averitec/development/baseline_scores.jsonl
  --source-candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl
  --protocol external --bootstrap-samples 5000 --seed 2026
)

python scripts/evaluate_v3_score_ensemble.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --baseline outputs/v3/climate_fever/full/conflict_audit_v1.jsonl \
  --candidate outputs/v3/climate_fever/full/conflicting_rescue_v1.jsonl \
  --meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/climate_fever \
  "${SOURCE_ENSEMBLE[@]}" \
  --output-dir outputs/v3/reviewer_controls/qwen_climate_fever/score_ensemble_v3

python scripts/evaluate_v3_score_ensemble.py \
  --gold data/processed/v3/scifact/dev_oracle.jsonl \
  --baseline outputs/v3/scifact/conflict_audit_v1.jsonl \
  --candidate outputs/v3/scifact/conflicting_rescue_v1.jsonl \
  --meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/scifact_oracle \
  "${SOURCE_ENSEMBLE[@]}" --project-conflicting-to-nei \
  --output-dir outputs/v3/reviewer_controls/qwen_scifact_oracle/score_ensemble_v3

python scripts/evaluate_v3_score_ensemble.py \
  --gold data/processed/v3/scifact/dev_retrieved_bm25_top5.jsonl \
  --baseline outputs/v3/scifact/retrieved_evidence/conflict_audit_v1.jsonl \
  --candidate outputs/v3/scifact/retrieved_evidence/conflicting_rescue_v1.jsonl \
  --meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/scifact_bm25 \
  "${SOURCE_ENSEMBLE[@]}" --project-conflicting-to-nei \
  --output-dir outputs/v3/reviewer_controls/qwen_scifact_bm25/score_ensemble_v3
```

Verify exact row counts before proceeding. If any assertion fails, do not run
the frontier with a substituted file from another protocol.

```bash
python - <<'PY'
from pathlib import Path
expected = {
    "outputs/v3/reviewer_controls/qwen_averitec/score_ensemble_v3/predictions.jsonl": 3068,
    "outputs/v3/reviewer_controls/qwen_climate_fever/score_ensemble_v3/predictions.jsonl": 1535,
    "outputs/v3/reviewer_controls/qwen_scifact_oracle/score_ensemble_v3/predictions.jsonl": 300,
    "outputs/v3/reviewer_controls/qwen_scifact_bm25/score_ensemble_v3/predictions.jsonl": 300,
}
for name, n in expected.items():
    path = Path(name)
    rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    print("OK" if rows == n else "BAD", rows, name)
    assert rows == n
PY
```

```bash
BUDGETS=(0 .05 .10 .1222 .15 .20 .30 .40 1.0)

python scripts/evaluate_v3_quality_budget_controls.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/averitec_grouped090 \
  --supervised-predictions outputs/v3/reviewer_round2/averitec/supervised_375/predictions.jsonl \
  --ensemble-predictions outputs/v3/reviewer_controls/qwen_averitec/score_ensemble_v3/predictions.jsonl \
  --budgets "${BUDGETS[@]}" \
  --output-dir outputs/v3/reviewer_round2/averitec/frontier

python scripts/evaluate_v3_quality_budget_controls.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --baseline outputs/v3/climate_fever/full/conflict_audit_v1.jsonl \
  --candidate outputs/v3/climate_fever/full/conflicting_rescue_v1.jsonl \
  --meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/climate_fever \
  --supervised-predictions outputs/v3/reviewer_round2/climate_fever/supervised/predictions.jsonl \
  --ensemble-predictions outputs/v3/reviewer_controls/qwen_climate_fever/score_ensemble_v3/predictions.jsonl \
  --budgets "${BUDGETS[@]}" \
  --output-dir outputs/v3/reviewer_round2/climate_fever/frontier

python scripts/evaluate_v3_quality_budget_controls.py \
  --gold data/processed/v3/scifact/dev_oracle.jsonl \
  --baseline outputs/v3/scifact/conflict_audit_v1.jsonl \
  --candidate outputs/v3/scifact/conflicting_rescue_v1.jsonl \
  --meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/scifact_oracle \
  --supervised-predictions outputs/v3/reviewer_round2/scifact_oracle/supervised/predictions.jsonl \
  --ensemble-predictions outputs/v3/reviewer_controls/qwen_scifact_oracle/score_ensemble_v3/predictions.jsonl \
  --project-conflicting-to-nei --budgets "${BUDGETS[@]}" \
  --output-dir outputs/v3/reviewer_round2/scifact_oracle/frontier

python scripts/evaluate_v3_quality_budget_controls.py \
  --gold data/processed/v3/scifact/dev_retrieved_bm25_top5.jsonl \
  --baseline outputs/v3/scifact/retrieved_evidence/conflict_audit_v1.jsonl \
  --candidate outputs/v3/scifact/retrieved_evidence/conflicting_rescue_v1.jsonl \
  --meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/scifact_bm25 \
  --supervised-predictions outputs/v3/reviewer_round2/scifact_bm25/supervised/predictions.jsonl \
  --ensemble-predictions outputs/v3/reviewer_controls/qwen_scifact_bm25/score_ensemble_v3/predictions.jsonl \
  --project-conflicting-to-nei --budgets "${BUDGETS[@]}" \
  --output-dir outputs/v3/reviewer_round2/scifact_bm25/frontier
```

Create one protocol-separated 4-by-2 accuracy/macro-F1 figure:

```bash
python scripts/plot_v3_quality_budget_frontier.py \
  --curve AVeriTeC=outputs/v3/reviewer_round2/averitec/frontier/quality_budget_curves.csv \
  --curve CLIMATE-FEVER=outputs/v3/reviewer_round2/climate_fever/frontier/quality_budget_curves.csv \
  --curve SciFact-Oracle=outputs/v3/reviewer_round2/scifact_oracle/frontier/quality_budget_curves.csv \
  --curve SciFact-BM25=outputs/v3/reviewer_round2/scifact_bm25/frontier/quality_budget_curves.csv \
  --output outputs/v3/figures/reviewer_round2/quality_cost_frontiers.pdf
```

Oracle curves are post-hoc ceilings, never deployable systems. Do not pool the
four panels into one benchmark ranking because their evidence protocols differ.

## 3. Per-backbone compatibility selection

The selector compares only already cross-fitted source predictions. For each
held-out fold, the compatibility rule is chosen on the other folds. The
all-source OOF choice is the deployable rule for frozen external evaluation.
Macro-F1 is the predeclared primary criterion; accuracy breaks ties.

```bash
python scripts/select_v3_compatibility_rule.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/averitec_grouped090 \
  --criterion macro_f1 --bootstrap-samples 5000 --seed 2026 \
  --output-dir outputs/v3/reviewer_round2/compatibility/qwen3_4b

SECOND_TAG=mistral7b_v03_c170c70
python scripts/select_v3_compatibility_rule.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --meta-dir outputs/v3/meta_routing_experiments/$SECOND_TAG/averitec_grouped090 \
  --criterion macro_f1 --bootstrap-samples 5000 --seed 2026 \
  --output-dir outputs/v3/reviewer_round2/compatibility/$SECOND_TAG

python - <<'PY'
import json
for path in (
    "outputs/v3/reviewer_round2/compatibility/qwen3_4b/summary.json",
    "outputs/v3/reviewer_round2/compatibility/mistral7b_v03_c170c70/summary.json",
):
    report = json.load(open(path, encoding="utf-8"))
    print(path)
    print("deployment:", report["all_source_deployment_choice"])
    print("fold choices:", [x["selected_rule"] for x in report["folds"]])
    print("crossfit:", report["cross_fitted_selection"])
PY
```

Selection agreement with the stronger same-population point estimate is a
procedural diagnostic, not independent evidence of generalization.

## 4. Return package

```bash
find outputs/v3/reviewer_round2 -type f \
  \( -name summary.json -o -name predictions.jsonl -o -name quality_budget_curves.csv \) \
  -print | sort

cat outputs/v3/reviewer_round2/{averitec,climate_fever,scifact_oracle,scifact_bm25}/supervised*/summary.json
cat outputs/v3/reviewer_round2/compatibility/*/summary.json
ls -lh outputs/v3/figures/reviewer_round2/quality_cost_frontiers.pdf
df -h /
```

Return the six `summary.json` files, four frontier CSVs, and the PDF. Paper
claims about class balance, Pareto dominance, or compatibility selection must
not be updated until these artifacts are inspected.
