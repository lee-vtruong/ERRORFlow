# ERRORFlow V3 reproducibility and leakage audit

Status: **WP0 COMPLETE — REPRODUCIBLE AND FOLD-ISOLATED, BUT TRAIN-SELECTED**.

## Scope and audit anchor

The audit was opened from `main` at commit `bc4aec2`. Historical artifacts are
not modified in place. The artifact-level auditor writes a new JSON result and
replays the materialized policy from cached B1, A3, and memory files.

## Code-provenance findings

The V3 implementation first appears at commit `41106e9` on 2026-10-12. The
router implementation already used defaults `top_k=5`, `min_support=2`, and
`cost_weight=0.1`, and routed only positive utility. The first bounded routing
results were committed later at `102a74c`. Thus these numeric defaults predate
the recorded V3 outcome inspection, although they were not included in the
original JSON configuration snapshot or selected by independent calibration.

`conflicting_rescue_v1` was advanced after a 150-example, transition-stratified
screen drawn from AVeriTeC train. The full 3,068-example action cache and R6
evaluation therefore reuse a train-selected action. No nested inner-fold action
or router selection was performed. The existing AVeriTeC R6 estimate must not
be called an untouched generalization estimate.

## Fold-isolation implementation

`analyze_v3_residuals.py` assigns exact-normalized-claim groups deterministically
to five outer folds. Every memory record stores its source outer fold. For an
AveriTeC evaluation query, `materialize_v3_utility_policy.py` excludes both its
claim ID and every memory record from its outer fold before estimating utility.
Gold labels are present in offline records but are not accessed by retrieval or
the route decision.

The full 3,068-entry memory is expected and valid for source-only SciFact
evaluation. For AVeriTeC evaluation, the same physical artifact may be loaded,
but only the four non-validation folds are eligible for each query. This
distinction must be verified from the actual server artifact before WP0 passes.

## Statistical implementation

- Accuracy and macro-F1 operate on exactly paired IDs in the significance CLI.
- Exact McNemar uses the two discordant correctness counts.
- Bootstrap resampling uses one shared sampled index vector for gold, B1, and
  R6, preserving pairs.
- The reported integer identity must satisfy `99 - 41 = 58` and
  `58 / 3068 = 0.0189048`, equal to the accuracy delta.
- Cached component latency is not a fresh end-to-end wall-clock measurement and
  must be labelled as estimated/cached when reported.

## Duplicate and evidence audit

Exact normalized duplicate claims are grouped by construction. The new auditor
also reports cross-fold token-Jaccard near duplicates, repeated normalized
evidence text, and shared source identifiers. Evidence/source overlap is
descriptive: common sources are not automatically leakage, especially when
multiple claims legitimately cite the same page.

## Current result classification

| Result | Preliminary classification | Reason |
|---|---|---|
| Historical S0 versus B1 on AVeriTeC train | `DESCRIPTIVE_ONLY` | Same development population; not a holdout comparison |
| R6 versus B1 on AVeriTeC | `VERIFIED_BUT_TRAIN_SELECTED` pending artifact checks | Fold exclusion is implemented, but A3 was train-screen selected and no nested selection was used |
| SciFact frozen source-only result | Separately audited external result | Source memory is allowed; interpretation depends on the predeclared label adapter |

No historical score is silently replaced. If artifact replay fails, the R6
classification changes to `INVALID_DUE_TO_LEAKAGE` or `UNVERIFIABLE`, and a new
run ID will be required.

## Server artifact audit command

Run only after pulling the audit implementation. This is CPU-only and reuses
all existing artifacts.

```bash
cd ~/whale/ERRORFlow
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python scripts/audit_v3_reproducibility.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --materialized outputs/v3/averitec/full_actions/r6_utility_predictions.jsonl \
  --reported-significance outputs/v3/averitec/full_actions/r6_utility_significance.json \
  --git-head "$(git rev-parse HEAD)" \
  --output outputs/v3/audits/averitec_v3_reproducibility.json
```

The gate passes only if `hard_checks_pass=true`, every fold has zero usable
validation-ID overlap, policy replay has zero route/prediction mismatches, and
recomputed statistics match the stored significance artifact.

## Correction log

- Opened audit; no historical artifact or score changed.
- Marked the AVeriTeC R6 estimate as train-selected rather than untouched OOF.
- Added artifact hashes, deterministic replay, target-gold invariance, duplicate
  checks, evidence overlap, metric reproduction, McNemar, and paired bootstrap.

## Server artifact audit result

The artifact audit completed at commit `a065427` in 123.7 seconds and returned
`hard_checks_pass=true`. It reproduced 3,068 examples, 357 routes, 99 helpful
and 41 harmful flips, net +58, accuracy delta +0.0189048, macro-F1 delta
+0.0154565, and 4.46545 average forward passes. The resulting classification
is `VERIFIED_BUT_TRAIN_SELECTED`.

All 60 exact normalized-claim duplicate groups remain within one fold. The
lexical audit nevertheless found 40 cross-fold claim pairs with normalized
set-token Jaccard at least 0.90. This is not automatically leakage: reordered
or templated claims can share tokens while expressing different propositions.
It does require a robustness analysis because memory similarity can exploit
near-identical wording.

The post-audit sensitivity evaluator rebuilds folds from connected components
at Jaccard thresholds 0.85, 0.90, and 0.95, rewrites fold provenance only in
new in-memory copies, replays R6, and writes new run IDs. Shared evidence text
or source identifiers are reported but are not used for grouping because doing
so could collapse legitimate multi-claim sources and create an unrealistic
domain split.

## Near-duplicate grouping result

All three conservative grouping thresholds preserve a positive result:

| Token-Jaccard threshold | Routes | Helpful | Harmful | Delta accuracy | Delta macro-F1 | McNemar p |
|---:|---:|---:|---:|---:|---:|---:|
| 0.85 | 336 | 86 | 41 | +0.01467 | +0.01198 | 8.06e-5 |
| 0.90 | 343 | 90 | 44 | +0.01499 | +0.01169 | 8.75e-5 |
| 0.95 | 336 | 92 | 35 | +0.01858 | +0.01432 | 4.36e-7 |

Every accuracy and macro-F1 bootstrap interval remains above zero. The strictest
macro-F1 lower bound occurs at threshold 0.90 and is still positive
(`0.000086`). Route-set Jaccard relative to the historical split is only
0.582–0.612, yet net correction remains positive. This shows the gain is not
explained by the 40 detected cross-fold near-duplicate pairs.

The artifact audit also reports 11 repeated normalized evidence texts and 171
shared source identifiers across folds. These are not direct router inputs:
memory retrieval uses claim text, B1 score profile, evidence length, and B1
prediction, but not evidence text, URL, title, or target gold. Grouping every
claim that shares a source would over-collapse legitimate multi-claim sources.
They are retained as a documented correlation risk rather than classified as
direct leakage.

## WP0 gate decision

- Direct target-ID/fold leakage: not found.
- Target-gold use in routing: not found; permutation invariance passed.
- Policy reproducibility: exact replay, zero route/prediction mismatches.
- Metric/statistic reproducibility: exact stored-result match.
- Exact and near-duplicate sensitivity: passed.
- Historical AVeriTeC R6 status: `VERIFIED_BUT_TRAIN_SELECTED`.
- Historical S0/B1 same-train comparison: `DESCRIPTIVE_ONLY`.

WP1 may proceed, but paper tables must retain the train-selection qualifier
unless a nested or genuinely independent confirmation replaces it.
