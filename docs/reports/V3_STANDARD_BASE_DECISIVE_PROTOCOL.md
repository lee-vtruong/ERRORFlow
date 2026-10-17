# ERRORFlow V3 — Decisive Standard-Base Outcome-Memory Experiment

## Purpose

This experiment asks whether outcome-based selective escalation still helps
when the starting verifier is the stronger, conventional four-label prompt.
It does **not** reuse predictions, outcomes, or memory from the historical
`conflict_audit_v1` / `conflicting_rescue_v1` pair.

The frozen pair is:

- **Standard Base:** `standard_verdict_v1`
- **Standard Alt:** `standard_deliberate_verdict_v1`

The alternative prompt performs a deliberate second-pass verification of
entities, scope, subclaims, missing information, contradiction, and mixed
evidence. Its prompt text and hash are frozen in `src/errorflow/v3_prompts.py`
before the full evaluation is inspected.

## Evaluation design

Run two independent grouped, five-fold source-domain evaluations:

1. AVeriTeC train (3,068 claims).
2. CLIMATE-FEVER official candidates (1,535 claims).

For every query, its complete near-duplicate component is assigned to one
outer fold. Memory records from that fold are excluded before retrieval. Gold
labels create Standard-Alt recovery/regression outcomes only in the other
folds; target gold is never a query-time routing feature.

The primary router is `Memory-Global-natural` with the predeclared operating
point: claim/label/evidence weights 0.35/0.50/0.15, top-k 5, minimum support 2,
cost weight 0.1, and utility threshold 0. Controls are Standard Base,
Standard Alt-All, same-label memory at the same route count, confidence at the
same route count, and random routing at the same route count over 20 seeds.

The hypothesis is supported on a dataset only if Memory-Global:

1. has higher accuracy than Standard Base;
2. has a strictly positive 95% near-duplicate-group bootstrap CI for the
   accuracy delta; and
3. uses fewer average model forward passes than Standard Alt-All.

This is a confirmatory grouped-development test, not an untouched held-out
estimate. Report both datasets regardless of direction. Do not revise the
prompt, router, or threshold after seeing one dataset.

## Server setup

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MODEL=/home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507
export ROOT=outputs/v3/standard_base_decisive

python -m pytest -q
```

## 1. Dry-run and smoke test

```bash
mkdir -p "$ROOT/smoke25"

python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v2/averitec/train_all.jsonl \
  --output "$ROOT/smoke25/standard_base.jsonl" \
  --action standard_verdict_v1 \
  --split train --seed 2026 --max-samples 25 --dry-run

python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v2/averitec/train_all.jsonl \
  --output "$ROOT/smoke25/standard_alt.jsonl" \
  --action standard_deliberate_verdict_v1 \
  --split train --seed 2026 --max-samples 25

wc -l "$ROOT/smoke25/standard_alt.jsonl"
cat "$ROOT/smoke25/standard_alt.manifest.json"
```

The dry run intentionally does not materialize Standard Base predictions. The
full commands below generate both members of the pair in a fresh directory.

## 2. AVeriTeC: generate the isolated Standard pair

```bash
mkdir -p "$ROOT/averitec/actions"

python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v2/averitec/train_all.jsonl \
  --output "$ROOT/averitec/actions/standard_base.jsonl" \
  --action standard_verdict_v1 \
  --split train --seed 2026

python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v2/averitec/train_all.jsonl \
  --output "$ROOT/averitec/actions/standard_alt.jsonl" \
  --action standard_deliberate_verdict_v1 \
  --split train --seed 2026

wc -l "$ROOT/averitec/actions/"*.jsonl
```

## 3. AVeriTeC: grouped cross-fit evaluation

```bash
python scripts/evaluate_v3_standard_pair.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --standard-base "$ROOT/averitec/actions/standard_base.jsonl" \
  --standard-alt "$ROOT/averitec/actions/standard_alt.jsonl" \
  --output-dir "$ROOT/averitec/crossfit" \
  --dataset AVeriTeC \
  --folds 5 --group-threshold 0.90 \
  --top-k 5 --min-support 2 --cost-weight 0.1 \
  --samples 5000 --seed 2026
```

## 4. CLIMATE-FEVER: generate the isolated Standard pair

```bash
mkdir -p "$ROOT/climate_fever/actions"

python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --output "$ROOT/climate_fever/actions/standard_base.jsonl" \
  --action standard_verdict_v1 \
  --split source_all --source-development --seed 2026

python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --output "$ROOT/climate_fever/actions/standard_alt.jsonl" \
  --action standard_deliberate_verdict_v1 \
  --split source_all --source-development --seed 2026

wc -l "$ROOT/climate_fever/actions/"*.jsonl
```

## 5. CLIMATE-FEVER: grouped cross-fit evaluation

```bash
python scripts/evaluate_v3_standard_pair.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --standard-base "$ROOT/climate_fever/actions/standard_base.jsonl" \
  --standard-alt "$ROOT/climate_fever/actions/standard_alt.jsonl" \
  --output-dir "$ROOT/climate_fever/crossfit" \
  --dataset Climate-FEVER \
  --folds 5 --group-threshold 0.90 \
  --top-k 5 --min-support 2 --cost-weight 0.1 \
  --samples 5000 --seed 2026
```

## 6. Integrity checks and compact export

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/v3/standard_base_decisive")
for dataset in ("averitec", "climate_fever"):
    directory = root / dataset / "crossfit"
    provenance = json.load(open(directory / "standard_pair_provenance.json", encoding="utf-8"))
    summary = json.load(open(directory / "summary.json", encoding="utf-8"))
    assert provenance["historical_audit_memory_allowed"] is False
    assert summary["evaluation"] == "decisive_standard_base_outcome_memory_crossfit"
    assert all(x["usable_validation_id_overlap"] == 0 for x in summary["fold_isolation"])
    print(dataset, provenance)
    print("hypothesis", summary["hypothesis_test"])
PY

python scripts/summarize_v3_standard_pair.py \
  "$ROOT/averitec/crossfit/summary.json" \
  "$ROOT/climate_fever/crossfit/summary.json" \
  | tee "$ROOT/RESULTS_FOR_CODEX.txt"

wc -l "$ROOT/averitec/crossfit/"*.jsonl "$ROOT/climate_fever/crossfit/"*.jsonl
df -h /
```

Send `outputs/v3/standard_base_decisive/RESULTS_FOR_CODEX.txt`. Also send both
`standard_pair_provenance.json` files if any integrity assertion fails.

## Interpretation rule

- **Both datasets pass:** the paper can claim that outcome-based selective
  escalation improves a stronger conventional Base under grouped in-domain
  evaluation while using less compute than Standard Alt-All.
- **Only one passes:** report dataset dependence; do not pool the evidence.
- **Neither passes:** retain Standard Base as the practical recommendation and
  narrow ERRORFlow to an audit-pair diagnostic result. Do not tune the new
  alternative prompt on these outcomes and rerun it as if confirmatory.
