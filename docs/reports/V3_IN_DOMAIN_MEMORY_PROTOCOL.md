# V3 in-domain outcome-memory protocol

This experiment tests the paper's source-memory-mismatch explanation without
changing the frozen Base or Alt prompts. It is deliberately separate from all
previous AVeriTeC-memory results.

Two estimates are produced:

1. **Grouped cross-fit:** every evaluation query excludes its complete
   near-duplicate group fold from memory. This is available for CLIMATE-FEVER
   and SciFact train.
2. **Frozen train-to-dev:** SciFact memory is built only from official train,
   then applied unchanged to dev under the same evidence and three-label
   protocol. Dev labels are used only after routes are materialized.

CLIMATE-FEVER has no independent official train split in the downloaded
release, so its grouped result is a source-domain cross-fitted estimate, not an
untouched held-out result. SciFact train and dev must use separately cached
three-label Base/Alt runs; do not mix the historical four-label projected cache
with the new three-label memory.

## 0. Setup and preflight

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MODEL=/home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507

python -m pytest -q
python scripts/evaluate_v3_in_domain_memory.py --help

for f in \
  data/processed/v3/climate_fever/all_official_candidates.jsonl \
  outputs/v3/climate_fever/full/conflict_audit_v1.jsonl \
  outputs/v3/climate_fever/full/conflicting_rescue_v1.jsonl \
  data/raw/scifact/data/corpus.jsonl \
  data/raw/scifact/data/claims_train.jsonl \
  data/raw/scifact/data/claims_dev.jsonl
do
  test -f "$f" && echo "OK $f" || echo "MISSING $f"
done

mkdir -p outputs/v3/in_domain_memory/{climate_fever,scifact/{oracle,bm25}} outputs/logs
df -h /
```

If the raw SciFact names differ, locate them before continuing:

```bash
find data/raw/scifact /home/stackops/whale/cache \
  -type f \( -name 'claims_train.jsonl' -o -name 'claims_dev.jsonl' -o -name 'corpus.jsonl' \) \
  -printf '%p\n' 2>/dev/null | sort
```

If `claims_train.jsonl` is absent everywhere, recover it from the official
AllenAI SciFact release without replacing the existing corpus or dev files:

```bash
SCIFACT_TMP=$(mktemp -d)
curl -fL \
  https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz \
  -o "$SCIFACT_TMP/data.tar.gz"

tar -tzf "$SCIFACT_TMP/data.tar.gz" | grep -E '(^|/)claims_(train|dev)\.jsonl$|(^|/)corpus\.jsonl$'
tar -xzf "$SCIFACT_TMP/data.tar.gz" -C "$SCIFACT_TMP"

CLAIMS_TRAIN=$(find "$SCIFACT_TMP" -type f -name claims_train.jsonl -print -quit)
test -n "$CLAIMS_TRAIN"
install -m 0644 "$CLAIMS_TRAIN" data/raw/scifact/data/claims_train.jsonl

wc -l data/raw/scifact/data/claims_train.jsonl
sha256sum data/raw/scifact/data/claims_train.jsonl
python - <<'PY'
import json
path = "data/raw/scifact/data/claims_train.jsonl"
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
print("rows:", len(rows))
print("unique_ids:", len({str(row["id"]) for row in rows}))
print("with_evidence:", sum(bool(row.get("evidence")) for row in rows))
assert len(rows) == len({str(row["id"]) for row in rows})
PY
```

The current official release is expected to contain 809 train claims. Stop if
the archive listing or uniqueness check fails; do not substitute a processed
or unlabeled split.

## 1. CLIMATE-FEVER grouped in-domain cross-fit

The Base and Alt caches already exist. This command builds outcomes, assigns
near-duplicate groups at threshold 0.90, excludes each query fold, and writes
all materialized controls.

```bash
set -o pipefail
python scripts/evaluate_v3_in_domain_memory.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --baseline outputs/v3/climate_fever/full/conflict_audit_v1.jsonl \
  --candidate outputs/v3/climate_fever/full/conflicting_rescue_v1.jsonl \
  --dataset Climate-FEVER \
  --output-dir outputs/v3/in_domain_memory/climate_fever/crossfit \
  --group-threshold 0.90 \
  --folds 5 \
  --samples 5000 \
  --seed 2026 \
  2>&1 | tee outputs/logs/in_domain_climate_fever_crossfit.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

## 2. Prepare SciFact train under matched evidence protocols

### 2.1 Oracle rationales

```bash
python scripts/prepare_scifact_oracle.py \
  --corpus data/raw/scifact/data/corpus.jsonl \
  --claims data/raw/scifact/data/claims_train.jsonl \
  --split train \
  --output data/processed/v3/scifact/train_oracle.jsonl \
  --manifest data/processed/v3/scifact/train_oracle_manifest.json
```

### 2.2 Open-corpus BM25

```bash
python scripts/prepare_scifact_retrieved.py \
  --corpus data/raw/scifact/data/corpus.jsonl \
  --claims data/raw/scifact/data/claims_train.jsonl \
  --split train \
  --output data/processed/v3/scifact/train_retrieved_bm25_top5.jsonl \
  --manifest data/processed/v3/scifact/train_retrieved_bm25_top5_manifest.json \
  --top-documents 5 \
  --top-sentences 5
```

## 3. Cache SciFact-train Base and Alt predictions

All four runs use the frozen three-label protocol. They may be resumed only
with the same manifest and model snapshot.

```bash
mkdir -p outputs/v3/in_domain_memory/scifact/{oracle,bm25}/train_actions

python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v3/scifact/train_oracle.jsonl \
  --output outputs/v3/in_domain_memory/scifact/oracle/train_actions/base.jsonl \
  --action conflict_audit_v1 \
  --label-protocol scifact3 \
  --split source_train \
  --source-development \
  --seed 2026

python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v3/scifact/train_oracle.jsonl \
  --output outputs/v3/in_domain_memory/scifact/oracle/train_actions/alt.jsonl \
  --action conflicting_rescue_v1 \
  --label-protocol scifact3 \
  --split source_train \
  --source-development \
  --seed 2026

python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v3/scifact/train_retrieved_bm25_top5.jsonl \
  --output outputs/v3/in_domain_memory/scifact/bm25/train_actions/base.jsonl \
  --action conflict_audit_v1 \
  --label-protocol scifact3 \
  --split source_train \
  --source-development \
  --seed 2026

python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v3/scifact/train_retrieved_bm25_top5.jsonl \
  --output outputs/v3/in_domain_memory/scifact/bm25/train_actions/alt.jsonl \
  --action conflicting_rescue_v1 \
  --label-protocol scifact3 \
  --split source_train \
  --source-development \
  --seed 2026
```

If a run is interrupted, repeat that exact command with `--resume`.

## 4. SciFact-train grouped cross-fit

```bash
python scripts/evaluate_v3_in_domain_memory.py \
  --gold data/processed/v3/scifact/train_oracle.jsonl \
  --baseline outputs/v3/in_domain_memory/scifact/oracle/train_actions/base.jsonl \
  --candidate outputs/v3/in_domain_memory/scifact/oracle/train_actions/alt.jsonl \
  --dataset SciFact \
  --output-dir outputs/v3/in_domain_memory/scifact/oracle/train_crossfit \
  --group-threshold 0.90 --folds 5 --samples 5000 --seed 2026

python scripts/evaluate_v3_in_domain_memory.py \
  --gold data/processed/v3/scifact/train_retrieved_bm25_top5.jsonl \
  --baseline outputs/v3/in_domain_memory/scifact/bm25/train_actions/base.jsonl \
  --candidate outputs/v3/in_domain_memory/scifact/bm25/train_actions/alt.jsonl \
  --dataset SciFact \
  --output-dir outputs/v3/in_domain_memory/scifact/bm25/train_crossfit \
  --group-threshold 0.90 --folds 5 --samples 5000 --seed 2026
```

## 5. Cache matched three-label SciFact-dev predictions

Historical four-label/projected outputs are not reused here because their
label-score vectors are incompatible with the train memory.

```bash
mkdir -p outputs/v3/in_domain_memory/scifact/{oracle,bm25}/dev_actions

python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v3/scifact/dev_oracle.jsonl \
  --output outputs/v3/in_domain_memory/scifact/oracle/dev_actions/base.jsonl \
  --action conflict_audit_v1 --label-protocol scifact3 \
  --split external_scifact_dev_oracle --external-evaluation --seed 2026

python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v3/scifact/dev_oracle.jsonl \
  --output outputs/v3/in_domain_memory/scifact/oracle/dev_actions/alt.jsonl \
  --action conflicting_rescue_v1 --label-protocol scifact3 \
  --split external_scifact_dev_oracle --external-evaluation --seed 2026

python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v3/scifact/dev_retrieved_bm25_top5.jsonl \
  --output outputs/v3/in_domain_memory/scifact/bm25/dev_actions/base.jsonl \
  --action conflict_audit_v1 --label-protocol scifact3 \
  --split external_scifact_dev_bm25 --external-evaluation --seed 2026

python scripts/run_v3_qwen_action.py \
  --model "$MODEL" \
  --input data/processed/v3/scifact/dev_retrieved_bm25_top5.jsonl \
  --output outputs/v3/in_domain_memory/scifact/bm25/dev_actions/alt.jsonl \
  --action conflicting_rescue_v1 --label-protocol scifact3 \
  --split external_scifact_dev_bm25 --external-evaluation --seed 2026
```

## 6. Frozen SciFact train-to-dev evaluation

Routes use only train memory. Do not pass `--project-conflicting-to-nei`: the
new runs already use the native three-label protocol.

```bash
python scripts/evaluate_v3_nlf_policy.py \
  --gold data/processed/v3/scifact/dev_oracle.jsonl \
  --baseline outputs/v3/in_domain_memory/scifact/oracle/dev_actions/base.jsonl \
  --candidate outputs/v3/in_domain_memory/scifact/oracle/dev_actions/alt.jsonl \
  --memory outputs/v3/in_domain_memory/scifact/oracle/train_crossfit/memory.jsonl \
  --output-dir outputs/v3/in_domain_memory/scifact/oracle/train_to_dev \
  --protocol external --samples 5000 --seed 2026

python scripts/evaluate_v3_nlf_policy.py \
  --gold data/processed/v3/scifact/dev_retrieved_bm25_top5.jsonl \
  --baseline outputs/v3/in_domain_memory/scifact/bm25/dev_actions/base.jsonl \
  --candidate outputs/v3/in_domain_memory/scifact/bm25/dev_actions/alt.jsonl \
  --memory outputs/v3/in_domain_memory/scifact/bm25/train_crossfit/memory.jsonl \
  --output-dir outputs/v3/in_domain_memory/scifact/bm25/train_to_dev \
  --protocol external --samples 5000 --seed 2026
```

## 7. Integrity checks and result export

```bash
python - <<'PY'
import json
from pathlib import Path

for path in Path("outputs/v3/in_domain_memory").glob("**/summary.json"):
    report = json.load(open(path, encoding="utf-8"))
    isolation = report.get("fold_isolation", [])
    assert all(row["usable_validation_id_overlap"] == 0 for row in isolation)
    print(path, "n=", report.get("n"), "isolation_rows=", len(isolation))
PY

python scripts/summarize_v3_in_domain_memory.py \
  outputs/v3/in_domain_memory/climate_fever/crossfit/summary.json \
  outputs/v3/in_domain_memory/scifact/oracle/train_crossfit/summary.json \
  outputs/v3/in_domain_memory/scifact/oracle/train_to_dev/summary.json \
  outputs/v3/in_domain_memory/scifact/bm25/train_crossfit/summary.json \
  outputs/v3/in_domain_memory/scifact/bm25/train_to_dev/summary.json \
  | tee outputs/v3/in_domain_memory/RESULTS_FOR_CODEX.txt

wc -l \
  outputs/v3/in_domain_memory/climate_fever/crossfit/{memory,predictions,neighbor_diagnostics}.jsonl \
  outputs/v3/in_domain_memory/scifact/{oracle,bm25}/train_crossfit/{memory,predictions,neighbor_diagnostics}.jsonl
df -h /
```

Send `outputs/v3/in_domain_memory/RESULTS_FOR_CODEX.txt`. The paper must not be
updated from partial manifests or from train cross-fit alone; the SciFact
train-to-dev result is the primary independent test of the mismatch account.
