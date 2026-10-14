# P0 mechanism control and qualitative case-study export

Both operations replay existing AVeriTeC artifacts and require no new model
inference. They preserve grouped fold exclusion at threshold 0.90.

## 1. Rerun N-B with the NB8 support/abstention control

NB8 keeps NB0's exact global top-five neighborhood, similarity denominator,
cost estimate, support count, and minimum-support decision. It sets only the
recovery/regression observations of cross-label records to zero. This avoids
the support loss in NB3 and directly tests whether cross-label *outcomes*, not
merely relaxed compatibility, change routing utility.

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
  --top-k 5 --min-support 2 --cost-weight 0.1 \
  --soft-label-penalty 0.5 --group-threshold 0.90 \
  --shuffle-seeds $(seq 2026 2045) \
  --bootstrap-samples 5000 --seed 2026 \
  2>&1 | tee outputs/logs/cross_label_ablation_nb8.log

echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
grep '^NB8_' outputs/v3/averitec/cross_label_ablation/RESULTS_FOR_CODEX.txt
```

Interpretation is deliberately conditional. A confirmed NB0 advantage over
NB8 supports useful cross-label outcome information. A null result does not
invalidate NLF, but narrows the claim to compatibility relaxation and
candidate-pool effects at the tested operating point.

## 2. Export two real worked cases

The exporter searches the frozen natural-threshold decisions for exactly:

1. B1 correct, A3 wrong, HardLabel routes, and NLF abstains;
2. B1 wrong, A3 correct, NLF routes, and HardLabel abstains.

Among qualifying cases it deterministically selects the largest utility
separation. Gold is used only for post-hoc qualitative selection, never by the
router. The JSON contains complete evidence and both fold-safe top-five
neighborhoods.

```bash
mkdir -p outputs/v3/averitec/worked_cases

python scripts/export_v3_worked_example.py \
  --examples data/processed/v2/averitec/train_all.jsonl \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --output-dir outputs/v3/averitec/worked_cases \
  --top-k 5 --min-support 2 --cost-weight 0.1 \
  --group-threshold 0.90 --seed 2026

cat outputs/v3/averitec/worked_cases/case_studies.md
```

Send both `case_studies.md` and `case_studies.json` back. The manuscript will
replace the current artifact contract with these observed cases only after
this export. If one category does not exist, the command deliberately exits
non-zero after writing diagnostics; do not substitute a synthetic example.

