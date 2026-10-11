# Step 25 — Full shared-task dev confirmation

The router threshold remains frozen at `0.90`. Prepare all 19,998 shared-task dev claims without gold evidence, retrieve deterministic OR evidence with resume support, then run baseline and the frozen cost-aware route.

This is development confirmation. `paper_dev` remains unopened for final evaluation until the policy and reporting protocol are locked.

## Retrieval result

- Completed rows: `19,998/19,998`.
- Claims with evidence: `19,998/19,998`.
- Runtime: `52,885.1s` (~14h 41m 25s).
- Mean throughput: approximately `0.378 claim/s`.
- Resume state: `resumed_from=0`; the complete artifact was produced in one session.
- Artifact: `data/processed/fever/shared_task_dev_all_retrieved.jsonl` (`22,425,161` bytes, approximately 22.4 MB).
- Full baseline artifact: `outputs/qwen3_baseline/shared_task_dev_all_confidence.jsonl`.
- Integrity: `19,998` rows, `19,998` unique IDs, zero duplicates, zero missing confidence.
- Baseline Accuracy: `0.5637063706`; Macro-F1: `0.5515256446`.
- Per-class F1: SUPPORTED `0.6515818711`, REFUTED `0.6035977926`, NEI `0.3993972700`.
- Baseline cost: `788.056 ms`, `31.9469` generated tokens and `1.0` LLM call per claim.
- Status: retrieval and frozen baseline are complete; threshold-`0.90` router evaluation remains.
- Frozen route subset: `1,181/19,998` claims (`5.91%`) satisfy `prediction=REFUTED AND confidence<=0.90`.
- Critic process was subsequently observed with PID `241809`, and prediction lines were present in terminal output. The process later exited while the critic artifact remained at `0` rows. This run is failed/incomplete and must not be merged. Inspect application log and kernel OOM events before rerunning.
