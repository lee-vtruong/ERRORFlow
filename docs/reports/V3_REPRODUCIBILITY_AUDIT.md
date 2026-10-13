# ERRORFlow V3 reproducibility and leakage audit

Status: **CODE-PROVENANCE AUDIT COMPLETE; SERVER ARTIFACT AUDIT PENDING**.

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
