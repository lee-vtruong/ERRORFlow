# ERRORFlow V3 M3b disagreement-aware learned arbitration

This is a final isolated experiment. It consumes the immutable outputs of the
M1/M2/M3 benchmark and writes only beneath `outputs/v3/m3b_learned_gate/`.
It does not replace or modify ERRORFlow V3, B1/A3 caches, memories, manifests,
or the paper.

## Fixed design

- Gate: dependency-free L2 logistic regression.
- Scope: HardLabel/NLF disagreement claims only; fixed-policy agreement is
  preserved exactly.
- Features (six only): `U_H`, `U_N`, `U_N-U_H`, cross-label similarity mass,
  B1 entropy, and neighborhood overlap.
- Target: one iff invoking A3 changes an incorrect B1 verdict into a correct
  verdict; zero iff invoking A3 damages a correct B1 verdict.
- Neutral disagreements are excluded from binary fitting and actionable
  decision accuracy, but retained in end-to-end routing evaluation.
- Source evaluation: nested group CV. Each outer-fold model selects L2 and its
  probability threshold on inner folds only, then predicts the untouched outer
  fold. The input features already exclude the query's complete grouped fold
  from memory; the evaluator rechecks neighbor-fold isolation.
- External evaluation: uses the frozen deployable source model. No target label
  is used for fitting, threshold selection, regularization, or routing.
- M2's cross-fitted fold-specific beta and M3a's possible action equivalence to
  Union are preserved as audited semantics, not treated as bugs.

## 0. Preflight

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
set +e
set -o pipefail

python -m pytest -q
python scripts/evaluate_v3_m3b_learned_gate.py --help
mkdir -p outputs/v3/m3b_learned_gate outputs/logs
```

All commands below are CPU-only because they reuse frozen prediction and
meta-feature artifacts.

## 1. Qwen source: AVeriTeC grouped OOF

```bash
QWEN_META=outputs/v3/meta_routing_experiments/qwen3_4b/averitec_grouped090
QWEN_M3B=outputs/v3/m3b_learned_gate/qwen3_4b/averitec_grouped090

python scripts/evaluate_v3_m3b_learned_gate.py \
  --source-meta-dir "$QWEN_META" \
  --target-meta-dir "$QWEN_META" \
  --output-dir "$QWEN_M3B" \
  --mode source-oof \
  --bootstrap-samples 5000 --seed 2026 \
  2>&1 | tee outputs/logs/m3b_qwen_averitec.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"

test -s "$QWEN_M3B/models.json" && echo QWEN_M3B_SOURCE_OK
```

## 2. Qwen external evaluations

### CLIMATE-FEVER

```bash
python scripts/evaluate_v3_m3b_learned_gate.py \
  --source-meta-dir "$QWEN_META" \
  --source-model-dir "$QWEN_M3B" \
  --target-meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/climate_fever \
  --output-dir outputs/v3/m3b_learned_gate/qwen3_4b/climate_fever \
  --mode external --bootstrap-samples 5000 --seed 2026 \
  2>&1 | tee outputs/logs/m3b_qwen_climate.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

### SciFact oracle

```bash
python scripts/evaluate_v3_m3b_learned_gate.py \
  --source-meta-dir "$QWEN_META" \
  --source-model-dir "$QWEN_M3B" \
  --target-meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/scifact_oracle \
  --output-dir outputs/v3/m3b_learned_gate/qwen3_4b/scifact_oracle \
  --mode external --bootstrap-samples 5000 --seed 2026 \
  2>&1 | tee outputs/logs/m3b_qwen_scifact_oracle.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

### SciFact BM25

```bash
python scripts/evaluate_v3_m3b_learned_gate.py \
  --source-meta-dir "$QWEN_META" \
  --source-model-dir "$QWEN_M3B" \
  --target-meta-dir outputs/v3/meta_routing_experiments/qwen3_4b/scifact_bm25 \
  --output-dir outputs/v3/m3b_learned_gate/qwen3_4b/scifact_bm25 \
  --mode external --bootstrap-samples 5000 --seed 2026 \
  2>&1 | tee outputs/logs/m3b_qwen_scifact_bm25.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

## 3. Mistral source and CLIMATE-FEVER

Mistral SciFact is intentionally omitted because no matching frozen Mistral
SciFact B1/A3 cache exists. Never mix Qwen target caches with Mistral memory or
models.

```bash
MISTRAL_TAG=mistral7b_v03_c170c70
MISTRAL_META=outputs/v3/meta_routing_experiments/$MISTRAL_TAG/averitec_grouped090
MISTRAL_M3B=outputs/v3/m3b_learned_gate/$MISTRAL_TAG/averitec_grouped090

python scripts/evaluate_v3_m3b_learned_gate.py \
  --source-meta-dir "$MISTRAL_META" \
  --target-meta-dir "$MISTRAL_META" \
  --output-dir "$MISTRAL_M3B" \
  --mode source-oof \
  --bootstrap-samples 5000 --seed 2026 \
  2>&1 | tee outputs/logs/m3b_mistral_averitec.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"

test -s "$MISTRAL_M3B/models.json" && echo MISTRAL_M3B_SOURCE_OK

python scripts/evaluate_v3_m3b_learned_gate.py \
  --source-meta-dir "$MISTRAL_META" \
  --source-model-dir "$MISTRAL_M3B" \
  --target-meta-dir outputs/v3/meta_routing_experiments/$MISTRAL_TAG/climate_fever \
  --output-dir outputs/v3/m3b_learned_gate/$MISTRAL_TAG/climate_fever \
  --mode external --bootstrap-samples 5000 --seed 2026 \
  2>&1 | tee outputs/logs/m3b_mistral_climate.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}"
```

## 4. Return results

Each run writes natural and M3b-matched-budget metrics, paired McNemar and
bootstrap tests, oracle regret, actionable-decision accuracy, feature
ablations, and all per-query predictions. Source runs additionally save every
fold model and the deployable source model in `models.json`.

```bash
find outputs/v3/m3b_learned_gate -type f \
  \( -name experiment_report.md -o -name audit.json \
     -o -name statistical_comparisons.json -o -name feature_ablation.csv \
     -o -name models.json \) -print | sort

find outputs/v3/m3b_learned_gate -type f -name experiment_report.md \
  -exec sh -c 'echo "===== $1 ====="; cat "$1"' _ {} \; \
  | tee outputs/v3/m3b_learned_gate/RESULTS_FOR_CODEX.txt

python -m pytest -q
df -h /
```

Decision rule for the paper: if M3b does not exceed the best fixed policy
consistently across source and external conditions, retain ERRORFlow V3
unchanged and report the learned arbitration result as a negative experiment.
