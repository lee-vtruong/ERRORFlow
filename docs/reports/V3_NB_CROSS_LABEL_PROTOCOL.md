# N-B: cross-label memory mechanism ablation

This study isolates what cross-label memories contribute to the final R6-NLF
router. It replays cached B1 and A3 predictions and therefore requires no Qwen
inference. All variants use grouped fold exclusion and remove the query claim
from memory.

## Frozen definitions

| ID | Definition |
|---|---|
| NB0 | Full R6-NLF top-5 retrieval |
| NB1 | HardLabel: same predicted label only |
| NB2 | Cross-label records only |
| NB3 | Retrieve the NB0 top-5, remove cross-label outcome contributions, do not backfill; fewer than two remaining records means unroutable |
| NB4 | Shuffle outcomes within source-fold/source-label strata and apply replacements only to records cross-label for the query |
| NB5 | Same-label priority; use cross-label records only if fewer than five same-label records remain |
| NB6 | Multiply cross-label similarity by 0.5 before top-5 selection |
| NB7 | Limit the top-5 to at most one or two cross-label records; quota two is primary and quota one is sensitivity |

NB4 uses seed 2026 for paired inference and seeds 2026--2045 for variability.
The shuffle keeps same-label outcomes and neighbor similarities unchanged.

## Run on the server

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python -m pytest -q
mkdir -p outputs/v3/averitec/cross_label_ablation outputs/logs
set -o pipefail

python scripts/evaluate_v3_cross_label_ablation.py \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --output-dir outputs/v3/averitec/cross_label_ablation \
  --top-k 5 \
  --min-support 2 \
  --cost-weight 0.1 \
  --soft-label-penalty 0.5 \
  --group-threshold 0.90 \
  --shuffle-seeds $(seq 2026 2045) \
  --bootstrap-samples 5000 \
  --seed 2026 \
  2>&1 | tee outputs/logs/cross_label_ablation.log

echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

## Integrity and result checks

```bash
python - <<'PY'
import json
from collections import Counter

root = "outputs/v3/averitec/cross_label_ablation"
r = json.load(open(f"{root}/summary.json", encoding="utf-8"))
rows = [json.loads(x) for x in open(f"{root}/mechanistic.jsonl", encoding="utf-8") if x.strip()]

print("evaluation:", r["evaluation"])
print("configuration:", r["configuration"])
print("primary_results:", len(r["primary_results"]))
print("comparisons:", len(r["matched_significance_nb0_minus_variant"]))
print("mechanism_summary:", r["mechanism_summary"])
print("mechanistic_rows:", len(rows))
print("unique_ids:", len({x["example_id"] for x in rows}))
print("cross_label_count@5:", Counter(x["cross_label_count_at_5"] for x in rows))
print("required_fields_present:", all({
    "cross_label_count@5",
    "cross_label_count_at_5",
    "cross_label_similarity_mass",
    "same_label_utility_contribution",
    "cross_label_utility_contribution",
    "route_changed_vs_hardlabel",
    "helpful_or_harmful_after_route",
} <= set(x) for x in rows))
PY

wc -l outputs/v3/averitec/cross_label_ablation/*.jsonl
cat outputs/v3/averitec/cross_label_ablation/RESULTS_FOR_CODEX.txt
```

Expected: 3,068 mechanistic rows, nine primary configurations (NB7 has quota
one and quota two), 18 natural/matched primary results, and eight matched
comparisons against NB0. Send `RESULTS_FOR_CODEX.txt` back for the paper update.

## Interpretation rules

- NB3 differs from HardLabel because it never backfills the NLF top-5 after
  suppressing cross-label records.
- A gain from NB0 over NB3 isolates the utility contribution of cross-label
  outcomes within the same retrieved neighborhood.
- A gain over NB4 supports cross-label instance--outcome association, not only
  access to a larger candidate pool.
- NB5 may coincide with HardLabel when at least five fold-safe same-label
  records exist; that equality is a valid mechanistic result.
- Natural and matched-budget results must both be reported. Holm correction is
  applied to the eight primary NB0-versus-control accuracy tests.

## Completed canonical result

The run completed with 3,068 grouped AVeriTeC examples and a matched NB0 route
count of 375. Cross-label records appear in the NLF top five for 339 queries
(11.05%) and for 123 of the 375 naturally routed queries (32.8%). Ninety
natural route decisions differ between NB0 and HardLabel.

| Variant | Natural routes | Natural accuracy | Natural Macro-F1 | Matched accuracy | Matched Macro-F1 | Holm accuracy p vs NB0 |
|---|---:|---:|---:|---:|---:|---:|
| NB0 Full NLF | 375 | .7585 | .5666 | .7585 | .5666 | -- |
| NB1 HardLabel | 343 | .7552 | .5577 | .7562 | .5622 | 1.000 |
| NB2 Cross-label only | 1,128 | .7317 | .5508 | .7379 | .5471 | 8.83e-7 |
| NB3 Cross-label suppression | 330 | .7585 | .5660 | .7581 | .5664 | 1.000 |
| NB4 Cross-label outcome shuffle, seed 2026 | 358 | .7568 | .5640 | .7565 | .5652 | 1.000 |
| NB5 Same-label priority | 343 | .7552 | .5577 | .7562 | .5622 | 1.000 |
| NB6 Soft penalty 0.5 | 343 | .7552 | .5577 | .7562 | .5622 | 1.000 |
| NB7 Quota 1 | 370 | .7559 | .5603 | .7555 | .5606 | .381 |
| NB7 Quota 2 | 371 | .7565 | .5639 | .7559 | .5634 | .270 |

NB4 is stable over 20 seeds: its matched accuracy is .7575 +/- .0009 and
matched Macro-F1 is .5663 +/- .0014 (population standard deviations). The
seed-2026 paired intervals against NB0 cross zero for both metrics.

The mechanism result is deliberately narrower than a claim that cross-label
outcomes are necessary. NB2 is strongly worse, so cross-label memory is not a
standalone substitute for same-label evidence. However, NB3 suppresses direct
cross-label outcome contributions while preserving the original NLF top-five
competition; it matches NB0 natural accuracy with 45 fewer routes, and its
matched differences are negligible. NB4 likewise does not show a confirmed
degradation. Thus the present evidence supports cross-label records as a
candidate-pool competition and selective-pruning mechanism, but does not
establish that their instance-specific corrective outcomes cause the gain.

NB5 and NB6 reproduce HardLabel exactly in this run: fold-safe same-label
memory is sufficient to fill top five, and a 0.5 penalty prevents cross-label
records from entering the selected neighborhood. Quotas interpolate between
HardLabel and NLF, but their multiplicity-corrected exact accuracy tests remain
inconclusive despite paired bootstrap intervals favoring NB0.
