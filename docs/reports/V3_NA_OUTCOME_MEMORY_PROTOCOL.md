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
