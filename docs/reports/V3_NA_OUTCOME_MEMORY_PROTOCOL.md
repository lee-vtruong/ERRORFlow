# N-A: direct R6-NLF outcome-memory ablation

NA0--NA8 are canonical paper-facing aliases of direct-NLF controls that have
already been evaluated. Do not rerun Qwen and do not recompute routes. The
materializer validates `compatibility_mode=none`, selects seed 2026 for paired
stochastic comparisons, retains 20-seed aggregates, and exports sample-level
predictions plus the NA7 budget sweep.

## Mapping

| NA ID | Frozen source control |
|---|---|
| NA0 Full R6-NLF | B9 Full Outcome Memory |
| NA1 No Memory | B0 No Memory |
| NA2 Random Neighbors | B1 Random Memory |
| NA3 Stratified Outcome Shuffle | B14 Stratified Outcome Shuffle |
| NA4 Global Outcome Prior | B6 Static Global |
| NA5 Class-Conditional Prior | B15 Class Conditional |
| NA6 Recovery-Only Utility | B10 Recovery Only |
| NA7 Error-Only Memory | B7 Error Only |
| NA8 Correct-Only Memory | B8 Correct Only |

For NA2, randomization replaces similarity ranking but retains similarity
values for outcome aggregation. Compatibility still enforces action, source
split, claim-ID exclusion, and grouped fold exclusion; it does not enforce the
historical predicted-label filter.

## Run on the server

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python -m pytest -q
mkdir -p outputs/v3/averitec/na_outcome_memory

python scripts/materialize_v3_nlf_outcome_ablation.py \
  --base-summary outputs/v3/averitec/memory_ablation_nlf/summary.json \
  --base-predictions outputs/v3/averitec/memory_ablation_nlf/predictions.jsonl \
  --control-summary outputs/v3/averitec/memory_controls_nlf/summary.json \
  --control-predictions outputs/v3/averitec/memory_controls_nlf/predictions.jsonl \
  --control-budget-predictions outputs/v3/averitec/memory_controls_nlf/budget_predictions.jsonl \
  --output-dir outputs/v3/averitec/na_outcome_memory \
  --seed 2026
```

Expected integrity fields:

```bash
python - <<'PY'
import json
r = json.load(open(
    "outputs/v3/averitec/na_outcome_memory/summary.json",
    encoding="utf-8",
))
print(r["evaluation"])
print(r["configuration"])
print("variants:", len(r["aliases"]))
print("results:", len(r["results"]))
print("significance:", len(r["matched_significance_na0_minus_control"]))
print("budget results:", len(r["budget_results"]))
PY

wc -l outputs/v3/averitec/na_outcome_memory/*.jsonl
cat outputs/v3/averitec/na_outcome_memory/RESULTS_FOR_CODEX.txt
```

Expected counts are 9 variants, 18 natural/matched results, 8 paired
comparisons, and 27 budget results (NA0/NA6/NA7 at nine budgets). Send
`RESULTS_FOR_CODEX.txt` back for the paper-facing NA table.

## Completed canonical result

The server materialization completed with a matched route count of 375
(12.22% of 3,068 examples). The table below reports the frozen seed-2026
matched-budget replay. NA2 and NA3 also have 20-seed aggregates; the paper
reports those means and keeps seed 2026 for paired inference.

| Variant | Helpful | Harmful | Net | Accuracy | Macro-F1 | Holm accuracy p vs NA0 |
|---|---:|---:|---:|---:|---:|---:|
| NA0 Full R6-NLF | 94 | 38 | +56 | .7585 | .5666 | -- |
| NA1 No memory | 0 | 0 | 0 | .7402 | .5460 | 7.12e-6 |
| NA2 Random neighbors | 27 | 38 | -11 | .7366 | .5432 | 3.81e-7 |
| NA3 Stratified outcome shuffle | 32 | 33 | -1 | .7399 | .5420 | 1.03e-5 |
| NA4 Global outcome prior | 13 | 27 | -14 | .7357 | .5427 | 4.00e-8 |
| NA5 Class-conditional prior | 65 | 25 | +40 | .7533 | .5453 | .609 |
| NA6 Recovery-only utility | 106 | 52 | +54 | .7578 | .5631 | 1.000 |
| NA7 Error-only memory | 107 | 55 | +52 | .7572 | .5630 | .734 |
| NA8 Correct-only memory | 5 | 9 | -4 | .7389 | .5442 | 1.54e-6 |

The result supports the value of outcome identity and neighbor retrieval:
NA0 has positive paired accuracy and macro-F1 intervals over NA1, NA2, NA3,
NA4, and NA8. It does not establish a matched-budget accuracy advantage over
NA5, NA6, or NA7. NA6 routes all 3,068 examples under its natural threshold
and produces net -13, whereas NA0 naturally routes 375 with net +56; the
regression term therefore has a clear abstention role even though the
fixed-budget difference is inconclusive.

Across the matched budget sweep, NA7 is slightly more accurate than NA0 at 5%
and 10%, NA0 is higher at 11.18%, they tie at 15%, and NA0 is higher from 20%
through 40%. NA6 is strongest through 20% but degrades below NA0 from 25%
onward. These are population--budget interactions, not evidence that one
reduced memory universally dominates the full outcome memory.
