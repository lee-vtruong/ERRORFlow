# R6-NLF frozen evaluation protocol

This protocol promotes **R6-NLF** (no hard baseline-label compatibility
filter) to the main ERRORFlow router after Group-C development on grouped
AVeriTeC. The action, memory, similarity weights, top-k, support threshold,
utility equation, and decision threshold remain unchanged. No external target
label may be used to alter the policy after these commands are run.

## Outputs produced for every setting

`evaluate_v3_nlf_policy.py` materializes:

- B1 and A3-All anchors;
- natural R6-HardLabel and R6-NLF routes;
- R6-HardLabel, R6-NLF, and Confidence-Only at the natural NLF budget;
- the same three routers at the natural HardLabel budget as sensitivity;
- Accuracy, Macro-F1, per-class F1, helpful/harmful/net, route rate, and
  average forward passes;
- exact McNemar and 5,000-sample paired-bootstrap comparisons;
- NLF cross-label neighbor counts and label-pair transitions;
- NLF-only, HardLabel-only, overlap, and exclusive-route outcomes;
- fresh CPU memory-lookup mean/p95 timing (not synchronized GPU latency);
- prediction-level, route-level, and neighbor-level JSONL artifacts.

The primary direct router comparison is `matched_nlf_budget_nlf_vs_hard`.
Natural mode is operational performance; matched mode isolates route quality
from route quantity. Confidence-Only is reported at both realized memory-router
budgets. It has no separately reported "natural" point because no confidence
threshold was predeclared before these target sets were opened; inventing one
now would turn an evaluation control into post-hoc target tuning.

## Server setup and preflight

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src

python -m pytest -q

for f in \
  data/processed/v2/averitec/train_all.jsonl \
  outputs/v3/averitec/development/baseline_scores.jsonl \
  outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  data/processed/v3/climate_fever/all_official_candidates.jsonl \
  outputs/v3/climate_fever/full/conflict_audit_v1.jsonl \
  outputs/v3/climate_fever/full/conflicting_rescue_v1.jsonl \
  data/processed/v3/scifact/dev_oracle.jsonl \
  outputs/v3/scifact/conflict_audit_v1.jsonl \
  outputs/v3/scifact/conflicting_rescue_v1.jsonl \
  data/processed/v3/scifact/dev_retrieved_bm25_top5.jsonl \
  outputs/v3/scifact/retrieved_evidence/conflict_audit_v1.jsonl \
  outputs/v3/scifact/retrieved_evidence/conflicting_rescue_v1.jsonl
do
  test -f "$f" && echo "OK $f" || echo "MISSING $f"
done

mkdir -p outputs/v3/nlf_eval outputs/logs
```

If preflight reports a missing SciFact BM25 prediction path, locate the cached
file before running; do not rerun model inference merely to rename it:

```bash
find outputs/v3/scifact -type f -name '*.jsonl' | sort
```

## 1. AVeriTeC grouped internal, threshold 0.90

```bash
set -o pipefail
python scripts/evaluate_v3_nlf_policy.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --output-dir outputs/v3/nlf_eval/averitec_grouped090 \
  --protocol grouped_internal \
  --group-threshold 0.90 \
  --samples 5000 \
  --seed 2026 \
  2>&1 | tee outputs/logs/r6_nlf_averitec_grouped090.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

## 2. CLIMATE-FEVER official candidates

```bash
set -o pipefail
python scripts/evaluate_v3_nlf_policy.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --baseline outputs/v3/climate_fever/full/conflict_audit_v1.jsonl \
  --candidate outputs/v3/climate_fever/full/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --output-dir outputs/v3/nlf_eval/climate_fever \
  --protocol external \
  --samples 5000 \
  --seed 2026 \
  2>&1 | tee outputs/logs/r6_nlf_climate_fever.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

## 3. SciFact oracle rationales

The predeclared adapter maps `CONFLICTING` to `NOT ENOUGH INFO` only after
routing. Raw pre-projection predictions remain in the output JSONL.

```bash
set -o pipefail
python scripts/evaluate_v3_nlf_policy.py \
  --gold data/processed/v3/scifact/dev_oracle.jsonl \
  --baseline outputs/v3/scifact/conflict_audit_v1.jsonl \
  --candidate outputs/v3/scifact/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --output-dir outputs/v3/nlf_eval/scifact_oracle \
  --protocol external \
  --project-conflicting-to-nei \
  --samples 5000 \
  --seed 2026 \
  2>&1 | tee outputs/logs/r6_nlf_scifact_oracle.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

## 4. SciFact open-corpus BM25

```bash
set -o pipefail
python scripts/evaluate_v3_nlf_policy.py \
  --gold data/processed/v3/scifact/dev_retrieved_bm25_top5.jsonl \
  --baseline outputs/v3/scifact/retrieved_evidence/conflict_audit_v1.jsonl \
  --candidate outputs/v3/scifact/retrieved_evidence/conflicting_rescue_v1.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --output-dir outputs/v3/nlf_eval/scifact_bm25 \
  --protocol external \
  --project-conflicting-to-nei \
  --samples 5000 \
  --seed 2026 \
  2>&1 | tee outputs/logs/r6_nlf_scifact_bm25.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

## Produce one result file to return to Codex

```bash
python scripts/summarize_v3_nlf_policy.py \
  outputs/v3/nlf_eval/averitec_grouped090/summary.json \
  outputs/v3/nlf_eval/climate_fever/summary.json \
  outputs/v3/nlf_eval/scifact_oracle/summary.json \
  outputs/v3/nlf_eval/scifact_bm25/summary.json \
  | tee outputs/v3/nlf_eval/RESULTS_FOR_CODEX.txt

wc -l outputs/v3/nlf_eval/*/predictions.jsonl \
  outputs/v3/nlf_eval/*/neighbor_diagnostics.jsonl
df -h /
```

Send `outputs/v3/nlf_eval/RESULTS_FOR_CODEX.txt`. The paper's pending external
R6-NLF cells must be updated only from these frozen artifacts.
