# Step 26 — Resumable and durable inference

Long inference jobs now support `--resume`. Existing claim IDs are validated and skipped, new records are appended, and every JSONL record is flushed immediately. Duplicate IDs, corrupt partial JSON and output IDs absent from the current input fail closed.

Use unbuffered Python, shell `pipefail`, append-mode logging and explicit whitespace before redirection:

```bash
set -o pipefail
python -u scripts/run_qwen3_baseline.py \
  --model /path/to/model \
  --input /path/to/input.jsonl \
  --output /path/to/predictions.jsonl \
  --split split_name \
  --intervention evidence_critic \
  --resume \
  2>&1 | tee -a /path/to/run.log
echo "PIPELINE_EXIT=$?"
```

Completion requires `INFERENCE_DONE`, exit code zero and output row count equal to input row count.
