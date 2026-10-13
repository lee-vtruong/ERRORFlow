# ERRORFlow V3 similarity-component ablation protocol

Status: **CODED; SERVER RESULTS PENDING**.

This experiment asks what residual error memory contributes beyond confidence,
similarity, and label-conditional routing, and which R6 similarity components
produce the observed gain. It is restricted to AVeriTeC train development
folds with token-Jaccard 0.90 grouped fold exclusion. No external target set is
used for tuning.

## Result inventory before this experiment

| Item | Existing result | Can substitute for A1--A8? |
|---|---|---|
| A0 Full R6 | Yes: grouped-primary accuracy 0.7552, macro-F1 0.5577, 343/3068 routes | Reference only |
| Confidence-only | Yes, matched-budget control | No: does not retrieve outcome memories |
| Label-conditional | Yes, matched-budget control | No: does not remove one R6 similarity component |
| Similarity-only | Yes, but uses the full composite similarity | No: not label-only, claim-only, or evidence-only |
| Random routing | Yes, matched-budget control | No: random claims differs from random compatible neighbors |
| A1--A8 | No component-specific result yet | Must run the new evaluator |

Existing numbers remain frozen in `V3_MATCHED_BUDGET_ABLATION.md`; they must
not be relabeled as component-removal results.

## Frozen variants

| ID | Similarity weights before normalization | Neighbor selection |
|---|---:|---|
| A0 Full R6 | 0.35 / 0.50 / 0.15 | ranked |
| A1 w/o claim | 0 / 0.50 / 0.15 | ranked |
| A2 w/o label score | 0.35 / 0 / 0.15 | ranked |
| A3 w/o evidence length | 0.35 / 0.50 / 0 | ranked |
| A4 label-score only | 0 / 1 / 0 | ranked |
| A5 claim only | 1 / 0 / 0 | ranked |
| A6 evidence-length only | 0 / 0 / 1 | ranked |
| A7 uniform | 1 / 1 / 1 | ranked |
| A8 random neighbors | 0.35 / 0.50 / 0.15 | deterministic random compatible top-5 |

For A8, neighbor selection is randomized but the selected records retain their
full similarity values when estimating recovery/regression. This isolates the
ranking effect. Twenty predeclared seeds (2026--2045) are aggregated.

For a single-feature variant, all selected similarities can legitimately be
zero (especially claim-only when no compatible memory shares a token). In
that case only, the estimator falls back to uniform weights over the selected
top-k records and reports `uniform_weight_fallback_count/rate`. Without this
declared fallback the weighted utility denominator is zero.

## Evaluation modes

1. **Natural:** route when the variant's own utility is greater than zero.
2. **Matched budget:** force exactly the A0 natural route count by selecting
   the highest-utility eligible claims for that variant.

Both modes report accuracy, macro-F1, per-class F1, helpful/harmful/net flips,
route rate, route overlap with A0, and forward passes. Deterministic
matched-budget variants receive paired A0-versus-variant McNemar and bootstrap
comparisons.

## Server command

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python -m pytest -q

python scripts/evaluate_v3_component_ablation.py \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --output-dir outputs/v3/ablations/component_removal_grouped090 \
  --group-threshold 0.90 \
  --top-k 5 \
  --min-support 2 \
  --cost-weight 0.1 \
  --utility-threshold 0.0 \
  --random-neighbor-seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040 2041 2042 2043 2044 2045
```

This uses cached artifacts and performs no LLM inference. Expected output:

```text
outputs/v3/ablations/component_removal_grouped090/summary.json
outputs/v3/ablations/component_removal_grouped090/routes.jsonl
outputs/v3/ablations/component_removal_grouped090/predictions.jsonl
```

## Return only the compact result

```bash
python - <<'PY'
import json

p = "outputs/v3/ablations/component_removal_grouped090/summary.json"
r = json.load(open(p, encoding="utf-8"))
print("matched_route_count:", r["configuration"]["matched_route_count"])
print("VARIANT\tMODE\tROUTED\tACC\tMACRO_F1\tD_ACC\tD_F1\tNET\tJACCARD_A0")
for key, x in r["results"].items():
    if "seed=" in key:
        continue
    variant, mode = key.split(":")
    overlap = x["route_overlap_with_a0_natural"]["jaccard"]
    print(variant, mode, x["routed"], f'{x["accuracy"]:.6f}',
          f'{x["macro_f1"]:.6f}', f'{x["delta_accuracy_vs_b1"]:.6f}',
          f'{x["delta_macro_f1_vs_b1"]:.6f}', x["net"],
          f'{overlap:.4f}', "fallback=" + str(x["uniform_weight_fallback_count"]), sep="\t")
print("A8_AGGREGATE")
print(json.dumps(r["random_neighbor_aggregate"], indent=2))
print("A8_VS_A0")
print(json.dumps(r["a8_matched_comparison_to_a0"], indent=2))
PY
```

Send that output back for interpretation and paper integration. Do not rerun or
tune A1--A8 on CLIMATE-FEVER or SciFact after seeing these development results.
