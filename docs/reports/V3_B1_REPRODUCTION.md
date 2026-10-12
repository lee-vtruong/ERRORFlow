# ERRORFlow V3 B1 reproduction

Status: **FULL TRAIN RUN COMPLETED**.

B1 is the immutable `conflict_audit_v1` prompt with mean four-label
log-probability scoring. V3 must reuse the cached train prediction file when it
already exists and passes ID/profile/hash validation. It must not rerun B1 only
to feed another router.

Expected train input: `data/processed/v2/averitec/train_all.jsonl` (`n=3068`).
The previous S0 train output is not B1 and cannot substitute for this run.

```bash
python scripts/run_v2_qwen_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input data/processed/v2/averitec/train_all.jsonl \
  --output outputs/v3/averitec/b1_train_all.jsonl \
  --label-set averitec --split train \
  --instruction-profile conflict_audit_v1 --decoding label-score
```

Use `--resume` only for the same output/configuration. Full-train metrics remain
unreported until the complete server execution.

## Smoke result

The bounded first-25 train run completed all rows in 10.40 seconds and recorded
exactly 100 model forward passes. The manifest validated:

- B1 prompt SHA-256 `700ebdc1...decf7`;
- input SHA-256 `f5d4c48e...0f3b`;
- four-label generation-config hash `89c39128...ad3`;
- 25 unique completed rows and no opened holdout input.

Smoke accuracy was 0.52 and macro-F1 0.45956. These numbers are functional
diagnostics on the first 25 examples, not representative performance estimates
and not a basis for prompt or threshold selection.

## Full train result

The hash-validated B1 cache completed all 3,068 official train examples in
364.65 seconds and recorded 12,272 forward passes. It achieved accuracy
0.74022 and macro-F1 0.54602, with 2,271 correct and 797 residual errors.

Relative to the historical S0 train baseline (accuracy 0.68644, macro-F1
0.50691), conflict-first B1 improves accuracy by 0.05378 and macro-F1 by
0.03911 while reducing the residual count from 962 to 797. This is a train-side
descriptive comparison, not independent generalization evidence.
