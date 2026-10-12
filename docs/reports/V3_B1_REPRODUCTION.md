# ERRORFlow V3 B1 reproduction

Status: **IMPLEMENTED; SERVER RUN NOT YET EXECUTED**.

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

Use `--resume` only for the same output/configuration. Actual metrics remain
unreported until server execution.
