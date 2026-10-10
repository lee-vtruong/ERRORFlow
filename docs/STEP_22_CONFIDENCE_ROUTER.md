# Step 22 — Confidence-aware selective router

Dev evaluation showed that routing every REFUTED prediction improves NEI F1 but harms REFUTED F1 and slightly lowers accuracy. The verifier now records `first_generated_token_max_probability` as a reproducible uncertainty proxy.

Rerun baseline inference to populate confidence. Then use `sweep_route_threshold.py` on train to select a threshold; freeze it before evaluating dev again. Do not select the threshold on `paper_dev`.
