# Group-C canonical rerun

The exploratory Group-C sweep used `\w+` claim tokenization, while the
established R6 memory implementation uses `[\w'-]+`. The runner now shares
the canonical boundary. This rerun evaluates only the six decision-relevant
configurations needed in the main paper, with the same grouped folds, memory,
action, and cached predictions.

## Configurations

- C0: R6-HardLabel reference.
- C2-4: Hellinger label-score similarity.
- C4-1: uniform component weights.
- C5-2: top-3 neighbors.
- C5-4: top-20 neighbors.
- C7-1: R6-NLF/no label filter.

Natural mode uses each configuration's own positive-utility boundary. Matched
mode uses the canonical C0 natural route count.

## Run on the server

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python -m pytest -q
mkdir -p outputs/v3/averitec/substitution_ablation/canonical_main outputs/logs

set -o pipefail
python scripts/evaluate_v3_substitution_ablation.py \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --examples data/processed/v2/averitec/train_all.jsonl \
  --output-dir outputs/v3/averitec/substitution_ablation/canonical_main \
  --variants \
    C2-4_hellinger \
    C4-1_uniform_sum \
    C5-2_top3 \
    C5-4_top20 \
    C7-1_no_label_filter \
  --group-threshold 0.90 \
  --bootstrap-samples 5000 \
  --seed 2026 \
  2>&1 | tee outputs/logs/group_c_canonical_main.log

echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

## Export results for the paper

```bash
python scripts/summarize_v3_substitution_ablation.py \
  outputs/v3/averitec/substitution_ablation/canonical_main/summary.json \
  | tee outputs/v3/averitec/substitution_ablation/canonical_main/RESULTS_FOR_CODEX.txt

wc -l outputs/v3/averitec/substitution_ablation/canonical_main/*.jsonl
cat outputs/v3/averitec/substitution_ablation/canonical_main/RESULTS_FOR_CODEX.txt
```

Return `RESULTS_FOR_CODEX.txt`. The exploratory full sweep remains in the
appendix; only this canonical subset may populate the main Group-C table.

## Next direct-final-method ablation

Groups A/B were run under the historical HardLabel policy. After this rerun,
the next confirmatory step is a compact canonical NLF memory ablation: no
memory, stratified outcome shuffle, class-conditional utility, recovery-only,
random neighbors, uniform weights, and full NLF. Do not reinterpret the old
HardLabel tables as direct component proofs for final NLF.
