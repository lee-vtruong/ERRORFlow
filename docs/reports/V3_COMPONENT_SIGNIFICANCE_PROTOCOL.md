# ERRORFlow V3 component-ablation paired significance

Status: **CODED; SERVER RESULTS PENDING**.

This post-hoc analysis consumes the completed A0--A8 component run. It does
not rerun retrieval or model inference. For each natural and matched-budget
mode it compares A0 Full R6 against A1--A8 using:

- prediction-level exact two-sided McNemar tests;
- 5,000-sample paired bootstrap intervals for accuracy and macro-F1 deltas;
- Holm correction over one family of eight A1--A8 McNemar tests;
- five-fold accuracy, macro-F1, and discordant-count analysis.

A8 uses seed 2026 as its predeclared primary member of the Holm family and
receives the full paired bootstrap. Seeds 2027--2045 are reported as
descriptive random-neighbor sensitivity only, without 19 redundant bootstrap
runs. Treating all 20 random seeds as independent hypotheses would
artificially enlarge the family.

## Run on the server

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python -m pytest -q

python scripts/analyze_v3_component_significance.py \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --component-summary outputs/v3/ablations/component_removal_grouped090/summary.json \
  --component-routes outputs/v3/ablations/component_removal_grouped090/routes.jsonl \
  --component-predictions outputs/v3/ablations/component_removal_grouped090/predictions.jsonl \
  --output-dir outputs/v3/ablations/component_removal_grouped090/paired_analysis \
  --a8-primary-seed 2026 \
  --bootstrap-samples 5000
```

Outputs:

```text
outputs/v3/ablations/component_removal_grouped090/paired_analysis/significance.json
outputs/v3/ablations/component_removal_grouped090/paired_analysis/paired_predictions.jsonl
```

## Compact output to return

```bash
python - <<'PY'
import json

p = "outputs/v3/ablations/component_removal_grouped090/paired_analysis/significance.json"
r = json.load(open(p, encoding="utf-8"))
print("REPLAY", r["prediction_replay"])
for mode, family in r["families"].items():
    print("\nMODE", mode)
    print("VARIANT\tA0_ONLY\tVARIANT_ONLY\tD_ACC\tD_F1\tRAW_P\tHOLM_P\tACC_CI\tF1_CI\tA0_BETTER_FOLDS_ACC\tA0_BETTER_FOLDS_F1")
    for name, x in family.items():
        b = x["paired_bootstrap_a0_minus_variant"]
        f = x["outer_fold_analysis"]
        print(name, x["a0_only_correct"], x["variant_only_correct"],
              f'{x["a0_minus_variant_accuracy"]:.6f}',
              f'{x["a0_minus_variant_macro_f1"]:.6f}',
              f'{x["mcnemar_exact_two_sided_p"]:.6g}',
              f'{x["holm_adjusted_mcnemar_p"]:.6g}',
              b["accuracy_delta_ci95"], b["macro_f1_delta_ci95"],
              f["a0_better_accuracy_folds"], f["a0_better_macro_f1_folds"], sep="\t")
print("\nA8 SENSITIVITY")
for mode, x in r["a8_seed_sensitivity"].items():
    print(mode, x["a0_minus_a8_accuracy"], x["a0_minus_a8_macro_f1"])
PY
```
