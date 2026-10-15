# ERRORFlow V3 experimental meta-routing benchmark

This protocol evaluates D0--D5, M1, M2, and the training-free M3a gate without
modifying the frozen V3 methods or paper. M3b is an optional trained comparator
and is deliberately not implemented in this dependency-free first benchmark.

## Integrity contract

- Canonical claim tokenizer: `[\w'-]+`, imported from
  `errorflow.error_memory.TOKEN_RE`. The curved-apostrophe form in the source
  request was text-encoding damage; it is not the frozen V3 implementation.
- Grouped source queries exclude their own claim and complete outer fold.
- External queries use source-only AVeriTeC memory.
- B1/A3 caches are immutable and are only selected between after routing.
- Gold and A3 predictions are unavailable to `errorflow.meta_routing.core`.
- The evaluator verifies B1/A3 model-manifest agreement, replays every memory
  outcome when source caches are supplied, and optionally requires exact
  HardLabel/NLF replay against a frozen summary.
- Qwen and Mistral must have separate source baseline, A3, and memory paths.

## Implemented policies

- D0: canonical HardLabel.
- D1: canonical NLF.
- D2: logical route union.
- D3: logical route intersection.
- D4: deterministic random arbitration in the disagreement region, 20 seeds.
- D5: offline accuracy oracle plus a separately labeled relaxed macro-F1
  ceiling. D5 is never exposed as an online route policy.
- M1: maximum canonical utility. The evaluator stops if it differs from D2.
- M2: maximum `U - beta * SE`, with weighted signed-outcome variance and
  effective sample size. `beta=0` must exactly reproduce M1.
- M3a: preserve fixed-policy agreement; in disagreement, accept the routing
  side only when its reliability-adjusted utility remains positive.

M2 beta is selected from `{0,.25,.5,1,1.5,2}` using only cross-fitted source
folds. External runs require the selected source beta explicitly.

## 0. Preflight

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
set +e
set -o pipefail

python -m pytest -q
python scripts/evaluate_v3_meta_routing.py --help
mkdir -p outputs/v3/meta_routing_experiments outputs/logs
```

## 1. Qwen source selection: grouped AVeriTeC

```bash
QWEN_META=outputs/v3/meta_routing_experiments/qwen3_4b/averitec_grouped090

python scripts/evaluate_v3_meta_routing.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --memory-baseline outputs/v3/averitec/development/baseline_scores.jsonl \
  --memory-candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --reference-summary outputs/v3/nlf_eval/averitec_grouped090/summary.json \
  --backbone-id qwen3_4b_instruct_2507 \
  --dataset-protocol averitec_grouped090 \
  --protocol grouped_internal \
  --group-threshold 0.90 --folds 5 \
  --budgets .05 .10 .15 .20 .30 \
  --bootstrap-samples 5000 --seed 2026 \
  --output-dir "$QWEN_META" \
  2>&1 | tee outputs/logs/meta_qwen_averitec.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"

QWEN_BETA=$(python -c "import json; print(json.load(open('$QWEN_META/method_configs.json'))['source_selected_deployable_beta'])")
QWEN_MAJORITY=$(python -c "import json; print(json.load(open('$QWEN_META/method_configs.json'))['source_selected_majority_action'])")
QWEN_CONFIDENCE=$(python -c "import json; print(json.load(open('$QWEN_META/method_configs.json'))['source_selected_confidence_threshold'])")
echo "QWEN_SOURCE_SELECTED_BETA=$QWEN_BETA"
echo "QWEN_SOURCE_CONTROLS=$QWEN_MAJORITY $QWEN_CONFIDENCE"
```

Do not continue if `audit_report.md` does not show `reference_replay: ok`,
unique memory IDs, fold exclusion, and exact M1/Union replay.

## 2. Qwen external replay

### CLIMATE-FEVER

```bash
python scripts/evaluate_v3_meta_routing.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --baseline outputs/v3/climate_fever/full/conflict_audit_v1.jsonl \
  --candidate outputs/v3/climate_fever/full/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --memory-baseline outputs/v3/averitec/development/baseline_scores.jsonl \
  --memory-candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --reference-summary outputs/v3/nlf_eval/climate_fever/summary.json \
  --backbone-id qwen3_4b_instruct_2507 \
  --dataset-protocol climate_fever_official_candidates \
  --protocol external --m2-beta "$QWEN_BETA" \
  --majority-action "$QWEN_MAJORITY" --confidence-threshold "$QWEN_CONFIDENCE" \
  --budgets .05 .10 .15 .20 .30 \
  --bootstrap-samples 5000 --seed 2026 \
  --output-dir outputs/v3/meta_routing_experiments/qwen3_4b/climate_fever \
  2>&1 | tee outputs/logs/meta_qwen_climate.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

### SciFact oracle

```bash
python scripts/evaluate_v3_meta_routing.py \
  --gold data/processed/v3/scifact/dev_oracle.jsonl \
  --baseline outputs/v3/scifact/conflict_audit_v1.jsonl \
  --candidate outputs/v3/scifact/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --memory-baseline outputs/v3/averitec/development/baseline_scores.jsonl \
  --memory-candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --reference-summary outputs/v3/nlf_eval/scifact_oracle/summary.json \
  --backbone-id qwen3_4b_instruct_2507 \
  --dataset-protocol scifact_oracle_projected \
  --protocol external --m2-beta "$QWEN_BETA" \
  --majority-action "$QWEN_MAJORITY" --confidence-threshold "$QWEN_CONFIDENCE" \
  --project-conflicting-to-nei \
  --budgets .05 .10 .15 .20 .30 \
  --bootstrap-samples 5000 --seed 2026 \
  --output-dir outputs/v3/meta_routing_experiments/qwen3_4b/scifact_oracle \
  2>&1 | tee outputs/logs/meta_qwen_scifact_oracle.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

### SciFact BM25

```bash
python scripts/evaluate_v3_meta_routing.py \
  --gold data/processed/v3/scifact/dev_retrieved_bm25_top5.jsonl \
  --baseline outputs/v3/scifact/retrieved_evidence/conflict_audit_v1.jsonl \
  --candidate outputs/v3/scifact/retrieved_evidence/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --memory-baseline outputs/v3/averitec/development/baseline_scores.jsonl \
  --memory-candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --reference-summary outputs/v3/nlf_eval/scifact_bm25/summary.json \
  --backbone-id qwen3_4b_instruct_2507 \
  --dataset-protocol scifact_bm25_projected \
  --protocol external --m2-beta "$QWEN_BETA" \
  --majority-action "$QWEN_MAJORITY" --confidence-threshold "$QWEN_CONFIDENCE" \
  --project-conflicting-to-nei \
  --budgets .05 .10 .15 .20 .30 \
  --bootstrap-samples 5000 --seed 2026 \
  --output-dir outputs/v3/meta_routing_experiments/qwen3_4b/scifact_bm25 \
  2>&1 | tee outputs/logs/meta_qwen_scifact_bm25.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

## 3. Mistral source selection and external replay

```bash
SECOND_TAG=mistral7b_v03_c170c70
MISTRAL_META=outputs/v3/meta_routing_experiments/$SECOND_TAG/averitec_grouped090

python scripts/evaluate_v3_meta_routing.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline outputs/v3/backbones/$SECOND_TAG/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/backbones/$SECOND_TAG/averitec/actions/a3.jsonl \
  --memory outputs/v3/backbones/$SECOND_TAG/averitec/actions/a3_memory.jsonl \
  --memory-baseline outputs/v3/backbones/$SECOND_TAG/averitec/development/baseline_scores.jsonl \
  --memory-candidate outputs/v3/backbones/$SECOND_TAG/averitec/actions/a3.jsonl \
  --reference-summary outputs/v3/backbones/$SECOND_TAG/averitec/eval/summary.json \
  --backbone-id "$SECOND_TAG" \
  --dataset-protocol averitec_grouped090 \
  --protocol grouped_internal \
  --group-threshold .90 --folds 5 \
  --budgets .05 .10 .15 .20 .30 \
  --bootstrap-samples 5000 --seed 2026 \
  --output-dir "$MISTRAL_META" \
  2>&1 | tee outputs/logs/meta_mistral_averitec.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"

MISTRAL_BETA=$(python -c "import json; print(json.load(open('$MISTRAL_META/method_configs.json'))['source_selected_deployable_beta'])")
MISTRAL_MAJORITY=$(python -c "import json; print(json.load(open('$MISTRAL_META/method_configs.json'))['source_selected_majority_action'])")
MISTRAL_CONFIDENCE=$(python -c "import json; print(json.load(open('$MISTRAL_META/method_configs.json'))['source_selected_confidence_threshold'])")
echo "MISTRAL_SOURCE_SELECTED_BETA=$MISTRAL_BETA"
echo "MISTRAL_SOURCE_CONTROLS=$MISTRAL_MAJORITY $MISTRAL_CONFIDENCE"

python scripts/evaluate_v3_meta_routing.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --baseline outputs/v3/backbones/$SECOND_TAG/climate_fever/actions/b1.jsonl \
  --candidate outputs/v3/backbones/$SECOND_TAG/climate_fever/actions/a3.jsonl \
  --memory outputs/v3/backbones/$SECOND_TAG/averitec/actions/a3_memory.jsonl \
  --memory-baseline outputs/v3/backbones/$SECOND_TAG/averitec/development/baseline_scores.jsonl \
  --memory-candidate outputs/v3/backbones/$SECOND_TAG/averitec/actions/a3.jsonl \
  --reference-summary outputs/v3/backbones/$SECOND_TAG/climate_fever/eval/summary.json \
  --backbone-id "$SECOND_TAG" \
  --dataset-protocol climate_fever_official_candidates \
  --protocol external --m2-beta "$MISTRAL_BETA" \
  --majority-action "$MISTRAL_MAJORITY" --confidence-threshold "$MISTRAL_CONFIDENCE" \
  --budgets .05 .10 .15 .20 .30 \
  --bootstrap-samples 5000 --seed 2026 \
  --output-dir outputs/v3/meta_routing_experiments/$SECOND_TAG/climate_fever \
  2>&1 | tee outputs/logs/meta_mistral_climate.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

Mistral SciFact is not run because matching Mistral SciFact B1/A3 caches do not
exist. Do not substitute Qwen caches.

## 4. Return checklist

```bash
find outputs/v3/meta_routing_experiments -type f \
  \( -name audit_report.md -o -name experiment_report.md \
     -o -name method_configs.json -o -name statistical_comparisons.json \) \
  -print | sort

find outputs/v3/meta_routing_experiments -type f -name experiment_report.md \
  -exec sh -c 'echo "===== $1 ====="; cat "$1"' _ {} \;

python -m pytest -q
df -h /
```

Every run directory contains all thirteen required files. Return the five
`experiment_report.md` files, five `oracle_headroom.json` files, and five
`statistical_comparisons.json` files for scientific interpretation. No result
is automatically adopted into ERRORFlow or the paper.
