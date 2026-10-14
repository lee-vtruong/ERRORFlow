# ERRORFlow V3 extended memory controls (B10--B17)

Status: **CODED; SERVER RESULTS PENDING**.

This experiment extends the completed B0--B9 study using only cached AVeriTeC
B1 and `conflicting_rescue_v1` predictions. It performs no new LLM inference.
All routing, calibration, and significance analysis remains inside grouped
AVeriTeC source folds; external evaluation labels are never used.

## Frozen definitions

| ID | Frozen implementation |
|---|---|
| B10 | Full memory; `recovery - 0.1 * cost` |
| B11 | Full memory; safety score `1 - regression - 0.1 * cost` |
| B12 | Uniform sample from full memory, size matched to B7 error-only memory; 20 seeds |
| B13 | Outcome-stratified sample from full memory at the same size; recovery/regression/neutral proportions retained approximately; 20 seeds |
| B14 | Outcome shuffle within `(outer_fold, B1 predicted label)`; 20 seeds |
| B15 | Static recovery/regression aggregate by B1 predicted label |
| B16 | Similarity-ranked top-k retrieval, but neighbors receive uniform weights when aggregating outcomes |
| B17 | Full-memory utility with a threshold selected separately for every target fold using only the other source folds |

B14 is a confirmatory replication of B4 under the leakage-safe definition.
B15 is mathematically the same control as B5. They remain materialized to make
the requested naming and replay checks explicit, but must not be counted as
new independent evidence.

The default B12/B13 size is inferred from B7. Pass `--size-match-count 797` to
assert the expected population explicitly. The script prints a warning if the
observed B7 size differs from 797.

## Statistical protocol

- B10--B17 use B9 as the reference.
- Primary stochastic replicate: seed 2026.
- B12--B14 additionally report mean, population standard deviation, minimum,
  and maximum over 20 seeds.
- Primary comparisons include paired bootstrap 95% intervals, exact two-sided
  McNemar tests, one Holm family across B10--B17, and outer-fold breakdowns.
- At every route budget, B7, B10, B14, and confidence-only are compared with B9
  at exactly the same number of routed claims. Holm correction is applied
  separately inside each budget family.
- Sample-level predictions are retained for every reported run.

## Run on the server

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python -m pytest -q

mkdir -p outputs/v3/ablations/memory_controls_grouped090

python scripts/evaluate_v3_memory_controls.py \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --output-dir outputs/v3/ablations/memory_controls_grouped090 \
  --group-threshold 0.90 \
  --top-k 5 \
  --min-support 2 \
  --cost-weight 0.1 \
  --utility-threshold 0.0 \
  --size-match-count 797 \
  --budgets 0.05 0.10 0.1118 0.15 0.20 0.25 0.30 0.35 0.40 \
  --bootstrap-samples 5000 \
  --random-seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040 2041 2042 2043 2044 2045 \
  2>&1 | tee outputs/v3/ablations/memory_controls_grouped090/run.log
```

Expected outputs:

```text
outputs/v3/ablations/memory_controls_grouped090/summary.json
outputs/v3/ablations/memory_controls_grouped090/routes.jsonl
outputs/v3/ablations/memory_controls_grouped090/predictions.jsonl
outputs/v3/ablations/memory_controls_grouped090/budget_predictions.jsonl
outputs/v3/ablations/memory_controls_grouped090/run.log
```

## Integrity checks and compact result

```bash
wc -l outputs/v3/ablations/memory_controls_grouped090/*.jsonl

python - <<'PY'
import json

p = "outputs/v3/ablations/memory_controls_grouped090/summary.json"
r = json.load(open(p, encoding="utf-8"))
c = r["configuration"]
print("MATCHED", c["matched_route_count"], c["matched_route_rate"])
print("SIZE", c["size_match_count"], c["error_memory_observed_count"])
print("VARIANT\tMODE\tSEED\tROUTED\tACC\tF1\tHELPFUL\tHARMFUL\tNET")
for key, x in r["results"].items():
    if "seed=" in key and not key.endswith("seed=2026"):
        continue
    variant, mode, *_ = key.split(":")
    print(variant, mode, x["seed"], x["routed"], f'{x["accuracy"]:.6f}',
          f'{x["macro_f1"]:.6f}', x["helpful"], x["harmful"], x["net"], sep="\t")

print("\nSTOCHASTIC")
print(json.dumps(r["stochastic_aggregate"], indent=2))

print("\nPRIMARY B9-MINUS-CONTROL")
for name, x in r["primary_matched_comparisons_b9_minus_variant"].items():
    boot = x["paired_bootstrap_reference_minus_variant"]
    print(name,
          "d_acc=", round(x["reference_minus_variant_accuracy"], 6),
          "d_f1=", round(x["reference_minus_variant_macro_f1"], 6),
          "raw_p=", x["mcnemar_exact"]["exact_two_sided_p"],
          "holm_p=", x["holm_adjusted_mcnemar_p"],
          "acc_ci=", boot["accuracy_delta_ci95"],
          "f1_ci=", boot["macro_f1_delta_ci95"])

print("\nBUDGETS")
for key, x in r["budget_results"].items():
    print(key, x["routed"], f'{x["accuracy"]:.6f}', f'{x["macro_f1"]:.6f}',
          x["helpful"], x["harmful"], x["net"])
PY
```

Return this output before adding any conclusions to the paper.
