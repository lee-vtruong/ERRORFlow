# SciFact native standard-prompt control and grouped uncertainty

This protocol addresses two separate reviewer checks without changing cached
Base/Alt outputs or the frozen method. First, it scores a conventional native
three-label SciFact prompt that contains no conflict-audit instruction.
Second, it materializes group-bootstrap intervals for the primary AVeriTeC
comparison. Results remain controls until inspected; they do not overwrite any
main prediction cache.

## Setup

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MODEL=/home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507
mkdir -p outputs/v3/reviewer_round4/{scifact_standard,grouped_uncertainty} outputs/logs
```

## Native three-label standard prompt

```bash
for item in \
  oracle_train:data/processed/v3/scifact/train_oracle.jsonl:source_train \
  oracle_dev:data/processed/v3/scifact/dev_oracle.jsonl:external_scifact_oracle_standard \
  bm25_train:data/processed/v3/scifact/train_retrieved_bm25_top5.jsonl:source_train \
  bm25_dev:data/processed/v3/scifact/dev_retrieved_bm25_top5.jsonl:external_scifact_bm25_standard
do
  IFS=: read -r name input split <<< "$item"
  mode=--external-evaluation
  [[ "$name" == *_train ]] && mode=--source-development
  python scripts/run_v3_qwen_action.py \
    --model "$MODEL" --input "$input" \
    --output "outputs/v3/reviewer_round4/scifact_standard/${name}.jsonl" \
    --action scifact_standard_v1 --label-protocol scifact3 \
    --split "$split" $mode --seed 2026
done
```

Evaluate the standard prompt against the existing native Base on dev:

```bash
for regime in oracle bm25; do
  if [ "$regime" = oracle ]; then GOLD=data/processed/v3/scifact/dev_oracle.jsonl; else GOLD=data/processed/v3/scifact/dev_retrieved_bm25_top5.jsonl; fi
  python scripts/evaluate_v3_significance.py \
    --gold "$GOLD" \
    --baseline "outputs/v3/in_domain_memory/scifact/$regime/dev_actions/base.jsonl" \
    --candidate "outputs/v3/reviewer_round4/scifact_standard/${regime}_dev.jsonl" \
    --output "outputs/v3/reviewer_round4/scifact_standard/${regime}_dev_vs_audit_base.json" \
    --evaluation-context external-source-only --samples 5000 --seed 2026
done
```

## Group-bootstrap primary AVeriTeC comparison

```bash
python scripts/evaluate_v3_grouped_pair.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/meta_routing_experiments/qwen3_4b/averitec_grouped090/per_query_predictions.jsonl \
  --candidate-filter policy=D1_NLF --candidate-filter mode=natural \
  --baseline-name Base --candidate-name Memory-Global \
  --group-threshold 0.90 --samples 5000 --seed 2026 \
  --output outputs/v3/reviewer_round4/grouped_uncertainty/averitec_base_vs_memory_global.json
```

For the outcome-balanced logistic comparison, both systems are in one cached
prediction file:

```bash
python scripts/evaluate_v3_grouped_pair.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline outputs/v3/meta_routing_experiments/qwen3_4b/averitec_grouped090/per_query_predictions.jsonl \
  --baseline-filter policy=D1_NLF --baseline-filter mode=natural \
  --candidate outputs/v3/reviewer_round2/averitec/supervised_375/predictions.jsonl \
  --candidate-filter method=Logistic-BalancedOutcome \
  --baseline-name Memory-Global --candidate-name Logistic-BalancedOutcome \
  --group-threshold 0.90 --samples 5000 --seed 2026 \
  --output outputs/v3/reviewer_round4/grouped_uncertainty/averitec_memory_vs_balanced_logistic.json
```

Return the two dev significance JSON files and both grouped-comparison JSON
files before changing the SciFact interpretation or the paper's confidence
intervals.
