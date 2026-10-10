# Step 23 — Frozen router variants

Train threshold sweep defines two variants before further dev evaluation:

- Quality-max: baseline prediction `REFUTED`, confidence threshold `1.0`; 290/995 routed; Accuracy `0.62613`, Macro-F1 `0.57279`.
- Cost-aware: baseline prediction `REFUTED`, confidence threshold `0.90`; 66/995 routed; Accuracy `0.61206`, Macro-F1 `0.55582`.

The cost-aware variant uses about 22.8% as many intervention calls as quality-max while retaining roughly half of its observed train gain. Both thresholds are frozen from train; do not retune them on dev or paper-dev.

The confidence value is `first_generated_token_max_probability`. It is an operational routing signal, not a calibrated probability of verdict correctness.
