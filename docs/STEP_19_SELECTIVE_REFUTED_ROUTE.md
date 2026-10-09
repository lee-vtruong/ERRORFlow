# Step 19 — Selective route by observable prediction

Route hypothesis: trigger `evidence_critic` only when the initial verifier predicts `REFUTED`. This avoids using gold-dependent error type at inference.

```bash
python scripts/build_route_subset.py \
  --claims data/processed/fever/train_first1000_retrieved.jsonl \
  --baseline outputs/qwen3_baseline/fever_train_first1000_retrieved.jsonl \
  --prediction REFUTED \
  --output data/processed/fever/train_predicted_refuted.jsonl
```

Run `evidence_critic` on this subset and compare its full accuracy against the original predictions for the same IDs. This is a train-side router experiment.
