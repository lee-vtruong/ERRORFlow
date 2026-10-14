# ERRORFlow V3 memory ablation protocol

Status: **CODED; SERVER RESULTS PENDING**.

This experiment tests whether outcome memory contributes more than static,
random, feature-only, or selection-biased memory. It uses the same AVeriTeC
grouped-0.90 development protocol, cached B1/A3 action, and fold exclusion as
the primary R6 result. No LLM inference or external target tuning is required.

## Frozen definitions

| ID | Definition |
|---|---|
| B0 | No memory and no correction; identical to B1 |
| B1 | Random compatible top-5 memories; 20 seeds |
| B2 | Claim-similarity nearest compatible top-5 |
| B3 | Label-score-similarity nearest compatible top-5 |
| B4 | Outcomes permuted within outer-fold and B1-label strata; 20 seeds |
| B5 | Uniform recovery/regression aggregate by B1 predicted label |
| B6 | Uniform global recovery/regression aggregate |
| B7 | Full retrieval using only source examples where B1 was wrong |
| B8 | Full retrieval using only source examples where B1 was correct |
| B9 | Full outcome memory (reference R6) |

B7 includes recovered and unrecovered errors. B8 includes regressed and
preserved correct controls. Their biased recovery/regression estimates are an
intended diagnostic result, not corrected away. B4 preserves fold/label
marginals while breaking the association between retrieval features and the
observed A3 outcome.

Natural mode applies `U>0`. Matched-budget mode uses exactly B9's natural route
count, except B0 which is definitionally never routed. B6 has no individual
ranking; matched-budget ties are resolved deterministically and must be
interpreted only as a static-global budget reference.

## Run on the server

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python -m pytest -q

python scripts/evaluate_v3_memory_ablation.py \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --output-dir outputs/v3/ablations/memory_grouped090 \
  --group-threshold 0.90 \
  --top-k 5 \
  --min-support 2 \
  --cost-weight 0.1 \
  --utility-threshold 0.0 \
  --random-seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2040 2041 2042 2043 2044 2045
```

Outputs:

```text
outputs/v3/ablations/memory_grouped090/summary.json
outputs/v3/ablations/memory_grouped090/routes.jsonl
outputs/v3/ablations/memory_grouped090/predictions.jsonl
```

## Compact result to return

```bash
python - <<'PY'
import json

p = "outputs/v3/ablations/memory_grouped090/summary.json"
r = json.load(open(p, encoding="utf-8"))
print("MATCHED", r["configuration"]["matched_route_count"], r["configuration"]["matched_route_rate"])
print("POPULATIONS", json.dumps(r["memory_populations"], indent=2))
print("VARIANT\tMODE\tSEED\tROUTED\tACC\tF1\tD_ACC\tD_F1\tNET\tJACCARD_B9")
for key, x in r["primary_results"].items():
    variant, mode, *rest = key.split(":")
    print(variant, mode, x["seed"], x["routed"], f'{x["accuracy"]:.6f}',
          f'{x["macro_f1"]:.6f}', f'{x["delta_accuracy_vs_b1"]:.6f}',
          f'{x["delta_macro_f1_vs_b1"]:.6f}', x["net"],
          f'{x["route_overlap_with_b9_natural"]["jaccard"]:.4f}', sep="\t")
print("STOCHASTIC")
print(json.dumps(r["stochastic_aggregate"], indent=2))
PY
```

Send this output back before any paper update. B4/B1 seed 2026 are the primary
replicates; the remaining seeds quantify sensitivity and must not be mixed
with a single-seed p-value without an explicit note.
