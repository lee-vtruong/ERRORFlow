# ERRORFlow V3 Group-C Substitution Ablation Protocol

## Purpose

This protocol evaluates whether the concrete design choices inside R6 matter,
without rerunning B1 or the corrective action. It replays cached prediction-level
artifacts under grouped 0.90 outer-fold exclusion. Target examples and every
memory record in their grouped outer fold are unavailable during routing.

The primary family is training-free and contains C1--C7 except C4-4 and C6-4.
C4-4 performs source-inner model selection. C6-4 fits a source-fold logistic
router and is therefore reported only as a trained-router comparator.

## Frozen definitions

- Reference: claim Jaccard + normalized-L1 label profile + evidence-token length.
- Weights: 0.35 / 0.50 / 0.15; top-k 5; minimum support 2.
- Utility: smoothed recovery minus regression minus 0.1 cost.
- Natural rule: route iff estimated utility is greater than zero.
- Matched budget: exactly the natural C0 route count.
- RRF constant: 60. Its normalized continuous fused score is also used as the
  utility-estimation neighbor weight.
- Soft label compatibility: mismatched B1 labels receive a 0.5 penalty.
- Fixed bins: confidence width 0.2 and margin width 0.1.
- Bayesian prior strength: 2.0, estimated from source-fold-eligible records.
- ESS smoothing pseudocount per outcome: `1 / sqrt(ESS)`.

Dense encoders are frozen as:

- Sentence-BERT: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- E5: `intfloat/e5-base-v2`, revision
  `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`; query/passage prefixes enabled.

## Server commands

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python -m pytest -q
python -c "import numpy, sentence_transformers; print('DENSE_DEPS_OK')"
```

Only if the dependency check fails:

```bash
pip install -e ".[v2]"
```

Create the two immutable embedding caches. This is the only GPU/model-download
stage in Group C; the ablation evaluator itself performs no LLM inference.

```bash
mkdir -p outputs/v3/averitec/substitution_ablation/embeddings

python scripts/precompute_v3_ablation_embeddings.py \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --examples data/processed/v2/averitec/train_all.jsonl \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --embedding-kind sentence_bert \
  --output outputs/v3/averitec/substitution_ablation/embeddings/minilm.npz \
  --device cuda

python scripts/precompute_v3_ablation_embeddings.py \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --examples data/processed/v2/averitec/train_all.jsonl \
  --model intfloat/e5-base-v2 \
  --model-revision f52bf8ec8c7124536f0efb74aca902b2995e5bcd \
  --embedding-kind e5 \
  --output outputs/v3/averitec/substitution_ablation/embeddings/e5_base_v2.npz \
  --device cuda
```

Run the complete training-free Group-C matrix. Progress is emitted every 250
queries for every variant. The output includes natural and matched-budget
predictions, routes, paired bootstrap intervals, exact McNemar tests, Holm
correction, and fold-level deltas.

```bash
mkdir -p outputs/logs outputs/v3/averitec/substitution_ablation/training_free
set -o pipefail

python scripts/evaluate_v3_substitution_ablation.py \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --examples data/processed/v2/averitec/train_all.jsonl \
  --sbert-cache outputs/v3/averitec/substitution_ablation/embeddings/minilm.npz \
  --e5-cache outputs/v3/averitec/substitution_ablation/embeddings/e5_base_v2.npz \
  --groups C1 C2 C3 C4 C5 C6 C7 \
  --output-dir outputs/v3/averitec/substitution_ablation/training_free \
  2>&1 | tee outputs/logs/v3_group_c_training_free.log

echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

Run P2 separately. It can be materially slower because C4-4 evaluates a fixed
source-inner weight grid independently for each outer fold.

```bash
mkdir -p outputs/v3/averitec/substitution_ablation/p2
set -o pipefail

python scripts/evaluate_v3_substitution_p2.py \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --examples data/processed/v2/averitec/train_all.jsonl \
  --reference-summary outputs/v3/averitec/substitution_ablation/training_free/summary.json \
  --reference-predictions outputs/v3/averitec/substitution_ablation/training_free/predictions.jsonl \
  --output-dir outputs/v3/averitec/substitution_ablation/p2 \
  2>&1 | tee outputs/logs/v3_group_c_p2.log

echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

## Integrity and result handoff

```bash
wc -l \
  outputs/v3/averitec/substitution_ablation/training_free/predictions.jsonl \
  outputs/v3/averitec/substitution_ablation/training_free/routes.jsonl \
  outputs/v3/averitec/substitution_ablation/p2/predictions.jsonl

python scripts/summarize_v3_substitution_ablation.py \
  outputs/v3/averitec/substitution_ablation/training_free/summary.json \
  outputs/v3/averitec/substitution_ablation/p2/summary.json \
  | tee outputs/v3/averitec/substitution_ablation/RESULTS_FOR_CODEX.txt
```

Send `RESULTS_FOR_CODEX.txt` back for interpretation and paper integration.
Do not tune any choice using SciFact or CLIMATE-FEVER.

## Outputs

- `summary.json`: metrics, significance, Holm correction, and protocol metadata.
- `predictions.jsonl`: sample-level predictions and routing decisions.
- `routes.jsonl`: selected IDs by variant and mode.
- `*.npz.manifest.json`: encoder identity, revision, inputs, and SHA-256 hashes.

## Interpretation constraints

- C4-1 duplicates A7 only if artifacts, grouping, and matched route count agree.
- C6-4 is not evidence that the training-free router itself improved.
- Natural-threshold and matched-budget conclusions must remain separate.
- A negative or null substitution result must be retained; substitutions are
  selected by mechanistic relevance, not by whether they favor R6.
