# V3 reviewer-remediation protocol

This protocol adds simple comparators and cost--quality controls without
changing ERRORFlow, its prompts, memories, cached B1/A3 predictions, or frozen
manifests. M3b is retained as a negative exploratory result and is not promoted
to the method.

## Interpretation boundary

AVeriTeC `dev` is **not an untouched holdout** in this repository. Rows 0--24
were used for smoke/development checks and rows 25--499 were opened during the
V2 holdout run. Any new V3 result on all 500 rows must be reported as
`post_selection_previously_opened_dev`. It can detect gross regression, but it
cannot provide an unbiased confirmatory estimate. A true held-out claim
requires the hidden official evaluation or another unopened dataset.

## 1. Score-level B1+A3 ensemble

This control always computes both prompts (eight label-scoring forward passes)
and mixes their normalized label-score distributions. On grouped AVeriTeC,
the mixture coefficient is selected on complementary outer folds. For an
external dataset, the coefficient is selected once on source AVeriTeC and then
frozen; target labels are used only for evaluation.

```bash
python scripts/evaluate_v3_score_ensemble.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline outputs/v3/averitec/full_actions/conflict_audit_v1.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/averitec_grouped090 \
  --protocol grouped_internal \
  --output-dir outputs/v3/reviewer_controls/qwen_averitec/score_ensemble
```

For CLIMATE-FEVER, use its target B1/A3 cache and freeze selection on the
source files above:

```bash
python scripts/evaluate_v3_score_ensemble.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --baseline outputs/v3/climate_fever/full/conflict_audit_v1.jsonl \
  --candidate outputs/v3/climate_fever/full/conflicting_rescue_v1.jsonl \
  --protocol external \
  --source-gold data/processed/v2/averitec/train_all.jsonl \
  --source-baseline outputs/v3/averitec/full_actions/conflict_audit_v1.jsonl \
  --source-candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --output-dir outputs/v3/reviewer_controls/qwen_climate_fever/score_ensemble
```

For SciFact, add `--project-conflicting-to-nei` and use the corresponding
oracle or BM25 B1/A3 cache.

## 2. Random, confidence, memory, and oracle quality--budget curves

The oracle is explicitly post hoc. `Oracle-at-most-budget` routes only helpful
actions up to the cap; `Oracle-exact-budget` fills the requested count and is
included only for exact-budget comparability. Random reports mean and standard
deviation over 20 declared seeds.

```bash
python scripts/evaluate_v3_quality_budget_controls.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline outputs/v3/averitec/full_actions/conflict_audit_v1.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/averitec_grouped090 \
  --output-dir outputs/v3/reviewer_controls/qwen_averitec/quality_budget \
  --budgets 0 .05 .10 .1222 .15 .20 .30 .40 .50 1.0
```

Repeat with each existing meta-routing directory. The script writes
`quality_budget_curves.csv`, `predictions.jsonl`, and `summary.json`, including
the fraction of post-hoc oracle helpful gain recovered by natural NLF.
For either SciFact protocol, append `--project-conflicting-to-nei` to match the
predeclared three-label adapter.

## 3. Previously opened AVeriTeC dev regression check

Normalize all 500 rows and run the frozen action prompts under a new output
namespace. Do not tune a threshold, compatibility mode, mixture coefficient,
or any other configuration on these labels.

```bash
python -m errorflow.cli normalize \
  --dataset averitec --split dev \
  --input data/raw/averitec/dev.json \
  --output data/processed/v3/averitec/dev_all_previously_opened.jsonl

python scripts/run_v3_qwen_action.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input data/processed/v3/averitec/dev_all_previously_opened.jsonl \
  --output outputs/v3/averitec_dev_postselection/actions/b1.jsonl \
  --action conflict_audit_v1 --split post_selection_previously_opened_dev --seed 2026

python scripts/run_v3_qwen_action.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input data/processed/v3/averitec/dev_all_previously_opened.jsonl \
  --output outputs/v3/averitec_dev_postselection/actions/a3.jsonl \
  --action conflicting_rescue_v1 --split post_selection_previously_opened_dev --seed 2026
```

Then run `evaluate_v3_meta_routing.py` with `--protocol external`, the frozen
AVeriTeC train memory, and this new B1/A3 pair. The paper must label the result
as post-selection evidence and must not call it held-out.

## 4. Reporting checklist

- Put the existing cross-fitted logistic router and static class-pattern
  control in the main comparison, not only the appendix.
- Report B1, A3-All, score ensemble, confidence, random, HardLabel, and NLF at
  the same realized budget.
- Report the post-hoc oracle as an upper bound, never as a deployable method.
- State when A3-All dominates a router and analyze source-memory conservatism.
- Treat HardLabel versus NLF as a backbone-dependent compatibility choice.
- Describe constant action cost as a threshold shift, not general cost-aware
  planning.
