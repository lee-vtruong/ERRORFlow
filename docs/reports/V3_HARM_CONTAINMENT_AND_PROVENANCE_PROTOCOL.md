# V3 harm-containment, provenance, and grouped-uncertainty protocol

This protocol adds descriptive safety accounting without claiming a formal
no-harm guarantee. It also directly compares CLIMATE-FEVER in-domain memory
with frozen AVeriTeC memory on the same claims, reruns grouped source
uncertainty, executes the already implemented balanced logistic controls, and
hashes the frozen Qwen snapshot byte-for-byte.

## 0. Setup

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
set -o pipefail

python -m pytest -q
mkdir -p outputs/v3/reviewer_round3/{harm_containment,memory_provenance,checksums} outputs/logs
```

## 1. Recompute grouped bootstrap for in-domain cross-fits

This reuses cached Base/Alt outputs. It does not call the language model.

```bash
python scripts/evaluate_v3_in_domain_memory.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --baseline outputs/v3/climate_fever/full/conflict_audit_v1.jsonl \
  --candidate outputs/v3/climate_fever/full/conflicting_rescue_v1.jsonl \
  --dataset Climate-FEVER \
  --output-dir outputs/v3/in_domain_memory/climate_fever/crossfit_groupboot \
  --group-threshold 0.90 --folds 5 --samples 5000 --seed 2026 \
  2>&1 | tee outputs/logs/in_domain_climate_groupboot.log

python scripts/evaluate_v3_in_domain_memory.py \
  --gold data/processed/v3/scifact/train_oracle.jsonl \
  --baseline outputs/v3/in_domain_memory/scifact/oracle/train_actions/base.jsonl \
  --candidate outputs/v3/in_domain_memory/scifact/oracle/train_actions/alt.jsonl \
  --dataset SciFact \
  --output-dir outputs/v3/in_domain_memory/scifact/oracle/train_crossfit_groupboot \
  --group-threshold 0.90 --folds 5 --samples 5000 --seed 2026

python scripts/evaluate_v3_in_domain_memory.py \
  --gold data/processed/v3/scifact/train_retrieved_bm25_top5.jsonl \
  --baseline outputs/v3/in_domain_memory/scifact/bm25/train_actions/base.jsonl \
  --candidate outputs/v3/in_domain_memory/scifact/bm25/train_actions/alt.jsonl \
  --dataset SciFact \
  --output-dir outputs/v3/in_domain_memory/scifact/bm25/train_crossfit_groupboot \
  --group-threshold 0.90 --folds 5 --samples 5000 --seed 2026
```

Each comparison now contains both `paired_bootstrap` (claim resampling) and
`grouped_paired_bootstrap` (complete near-duplicate-group resampling). Use the
latter for the revised source-domain confidence intervals.

## 2. Direct CLIMATE-FEVER memory-provenance comparison

This is the missing paired test: the two routers are compared on exactly the
same 1,535 claims. Positive deltas mean in-domain memory is better.

```bash
python scripts/compare_v3_memory_provenance.py \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --in-domain-predictions outputs/v3/in_domain_memory/climate_fever/crossfit_groupboot/predictions.jsonl \
  --in-domain-policy Memory-Global-natural \
  --source-memory-predictions outputs/v3/meta_routing_experiments/qwen3_4b/climate_fever/per_query_predictions.jsonl \
  --source-memory-policy D1_NLF --source-memory-mode natural \
  --group-threshold 0.90 --folds 5 --samples 5000 --seed 2026 \
  --output outputs/v3/reviewer_round3/memory_provenance/climate_in_domain_vs_averitec.json \
  --rows-output outputs/v3/reviewer_round3/memory_provenance/climate_in_domain_vs_averitec.jsonl
```

Do not claim that memory mismatch is confirmed unless the direct group-level
interval excludes zero in the in-domain direction.

## 3. Harm avoided and helpful gain retained

```bash
python scripts/summarize_v3_harm_containment.py \
  --name qwen_averitec \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --router-predictions outputs/v3/meta_routing_experiments/qwen3_4b/averitec_grouped090/per_query_predictions.jsonl \
  --router-policy D1_NLF --router-mode natural \
  --output outputs/v3/reviewer_round3/harm_containment/qwen_averitec.json

SECOND_TAG=mistral7b_v03_c170c70
python scripts/summarize_v3_harm_containment.py \
  --name mistral_averitec \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline outputs/v3/backbones/$SECOND_TAG/averitec/actions/b1.jsonl \
  --candidate outputs/v3/backbones/$SECOND_TAG/averitec/actions/a3.jsonl \
  --router-predictions outputs/v3/meta_routing_experiments/$SECOND_TAG/averitec_grouped090/per_query_predictions.jsonl \
  --router-policy D1_NLF --router-mode natural \
  --output outputs/v3/reviewer_round3/harm_containment/mistral_averitec.json

python scripts/summarize_v3_harm_containment.py \
  --name climate_fever_in_domain \
  --gold data/processed/v3/climate_fever/all_official_candidates.jsonl \
  --baseline outputs/v3/climate_fever/full/conflict_audit_v1.jsonl \
  --candidate outputs/v3/climate_fever/full/conflicting_rescue_v1.jsonl \
  --router-predictions outputs/v3/in_domain_memory/climate_fever/crossfit_groupboot/predictions.jsonl \
  --router-policy Memory-Global-natural \
  --output outputs/v3/reviewer_round3/harm_containment/climate_in_domain.json

for regime in oracle bm25; do
  if [ "$regime" = oracle ]; then GOLD=data/processed/v3/scifact/dev_oracle.jsonl; else GOLD=data/processed/v3/scifact/dev_retrieved_bm25_top5.jsonl; fi
  python scripts/summarize_v3_harm_containment.py \
    --name scifact_${regime}_native \
    --gold "$GOLD" \
    --baseline outputs/v3/in_domain_memory/scifact/$regime/dev_actions/base.jsonl \
    --candidate outputs/v3/in_domain_memory/scifact/$regime/dev_actions/alt.jsonl \
    --router-predictions outputs/v3/in_domain_memory/scifact/$regime/train_to_dev/predictions.jsonl \
    --router-policy R6-NLF --router-mode natural \
    --output outputs/v3/reviewer_round3/harm_containment/scifact_${regime}_native.json
done
```

The output defines harm avoided as `1 - router_harmful / Alt_harmful` and gain
retained as `router_helpful / Alt_helpful`. Report both underlying counts.

## 4. Balanced logistic controls

Run Section 1 of `docs/reports/V3_REVIEWER_ROUND2_PROTOCOL.md`. The required
AVeriTeC outputs are `Logistic-BalancedOutcome` and
`Logistic-PerLabelRank`, including 375-route helpful/harmful counts, per-class
F1, exact McNemar tests, Holm p-values, and familywise intervals. Do not retain
the old abstract caveat after these artifacts have been inspected.

## 5. Byte-level Qwen checksum

This streams the model files and writes only a small JSON manifest; it does not
duplicate the checkpoint.

```bash
python scripts/hash_model_snapshot.py \
  --model-dir /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --output outputs/v3/reviewer_round3/checksums/qwen3_4b_snapshot.json \
  2>&1 | tee outputs/logs/qwen3_4b_sha256.log
```

If the schema-v1 manifest has already been produced, upgrade its aggregate
fields without rereading the 8 GB checkpoint:

```bash
python scripts/hash_model_snapshot.py \
  --model-dir /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --output outputs/v3/reviewer_round3/checksums/qwen3_4b_snapshot.json \
  --reuse-existing
```

Use `weights_sha256` as the stable weight digest. `snapshot_sha256` also
covers tokenizer/configuration artifacts but excludes mutable `.cache`
downloader metadata; `full_tree_sha256` is retained only as an audit value.

## 6. Return package

```bash
python - <<'PY'
import glob, json
for path in sorted(glob.glob("outputs/v3/reviewer_round3/**/*.json", recursive=True)):
    report = json.load(open(path, encoding="utf-8"))
    print("\n===", path, "===")
    if "harm_containment" in report:
        print(report["harm_containment"])
    elif "comparison" in report:
        print(report["comparison"])
    else:
        print({k: report.get(k) for k in ("snapshot_sha256", "weights_sha256", "total_bytes")})
PY

df -h /
```

Return the printed output plus the grouped-bootstrap comparison blocks from
the three new in-domain `summary.json` files.

## Recorded findings (2026-10-17)

- On 1,535 CLIMATE-FEVER claims, in-domain memory scores 0.5440 accuracy and
  0.4143 macro-F1 versus 0.5368/0.4117 for frozen AVeriTeC memory. The direct
  differences are +0.00717/+0.00260, but McNemar p=0.329 and both claim-level
  and grouped 95% intervals cross zero. This does not confirm memory mismatch.
- Qwen AVeriTeC Alt-All has 192 helpful and 205 harmful flips; Memory-Global
  retains 94 helpful flips and exposes 38 harmful flips, corresponding to
  49.0% gain retained and 81.5% harm avoided.
- The initial checksum manifest included mutable `.cache/huggingface` files.
  Schema v2 separates the stable weight digest from downloader audit state.
  For the three reported Qwen safetensors shards, `weights_sha256` is
  `a36b5737652a4f20b2a89b6d692d7ac939ac9f1f4da755864e9b49f56ec094d9`.
- Mistral, CLIMATE-FEVER in-domain, and native SciFact harm-containment JSON
  files remain required before extending the manuscript's audited-count table.
