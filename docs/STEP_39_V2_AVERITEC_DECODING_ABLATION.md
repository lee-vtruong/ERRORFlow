# Step 39 — AVeriTeC prompt and decoding ablation

## Finding from Step 38

All 25 generated outputs reached the 32-token cap. The label appeared first in
every inspected error, so truncation did not break label parsing; it wasted
tokens and cut explanations. The plain prompt never selected CONFLICTING because
it listed labels without defining the AVeriTeC distinction.

## Controlled comparison

Keep the same model, 25 examples, gold evidence, and four-label space. Compare:

1. `G1`: explicit AVeriTeC definitions, generation capped at 8 tokens.
2. `S1`: the same definitions, no generation; select the highest mean
   conditional log-probability among the four complete label strings.

The historical `G0` result is accuracy 0.64, macro-F1 0.49485, and 32 generated
tokens/example.

## Server commands

```bash
python scripts/run_v2_qwen_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input data/processed/v2/averitec/dev_first25.jsonl \
  --output outputs/v2/averitec/oracle_dev_first25_g1.jsonl \
  --label-set averitec --split dev_first25 \
  --instruction-profile averitec_v1 --decoding generate --max-new-tokens 8

python scripts/evaluate_v2_predictions.py \
  --gold data/processed/v2/averitec/dev_first25.jsonl \
  --predictions outputs/v2/averitec/oracle_dev_first25_g1.jsonl \
  --output outputs/v2/averitec/oracle_dev_first25_g1_metrics.json

python scripts/run_v2_qwen_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input data/processed/v2/averitec/dev_first25.jsonl \
  --output outputs/v2/averitec/oracle_dev_first25_s1.jsonl \
  --label-set averitec --split dev_first25 \
  --instruction-profile averitec_v1 --decoding label-score

python scripts/evaluate_v2_predictions.py \
  --gold data/processed/v2/averitec/dev_first25.jsonl \
  --predictions outputs/v2/averitec/oracle_dev_first25_s1.jsonl \
  --output outputs/v2/averitec/oracle_dev_first25_s1_metrics.json
```

## Selection rule

Prefer the protocol with better macro-F1, provided accuracy does not materially
decrease. Also inspect CONFLICTING recall and latency. Do not select from accuracy
alone because the 25-example slice contains only three CONFLICTING examples.

## Result

Completed on the 25-example calibration slice.

| Protocol | Accuracy | Macro-F1 | Latency/example | Generated tokens |
|---|---:|---:|---:|---:|
| G0 plain generation (32) | 0.64 | 0.49485 | 807.0 ms | 32 |
| G1 defined generation (8) | 0.60 | 0.47639 | 249.5 ms | 8 |
| S1 defined label score | 0.60 | 0.47639 | 150.3 ms | 0 |
| S0 plain label score | **0.64** | **0.49762** | **154.6 ms** | **0** |

S0 is selected. It matches G0 accuracy, slightly improves macro-F1, and reduces
latency by roughly 5.2x. It uses four model forward passes per example, which is
now reported explicitly despite generating zero output tokens. None of the four
protocols correctly predicted a CONFLICTING example, so that remains a known
verifier weakness rather than an ErrorFlow gain.
