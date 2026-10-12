# Step 32 — Screen B1 interventions on baseline errors

## Hypothesis

B1 errors are dominated by high-confidence NEI overcommitment, not low confidence or one retrieval source. The new `entailment_gate` explicitly distinguishes topical relevance from direct support/contradiction and forbids filling missing claim links with assumptions.

This is a counterfactual development experiment on the 417 baseline errors. It does not establish deployment benefit until the selected action is also tested on the 583 baseline-correct controls.

## Run both candidate actions

```bash
python scripts/run_qwen3_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input outputs/qwen3_baseline/shared_task_dev_first1000_v1_errors.jsonl \
  --output outputs/qwen3_interventions/v1_errors_evidence_critic.jsonl \
  --split shared_task_dev_first1000_v1_errors \
  --intervention evidence_critic

python scripts/run_qwen3_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input outputs/qwen3_baseline/shared_task_dev_first1000_v1_errors.jsonl \
  --output outputs/qwen3_interventions/v1_errors_entailment_gate.jsonl \
  --split shared_task_dev_first1000_v1_errors \
  --intervention entailment_gate
```

Both runners support `--resume` after interruption.

## Evaluate recovery

```bash
python scripts/evaluate_intervention.py \
  --errors outputs/qwen3_baseline/shared_task_dev_first1000_v1_errors.jsonl \
  --intervention outputs/qwen3_interventions/v1_errors_evidence_critic.jsonl \
  --output outputs/qwen3_interventions/v1_errors_evidence_critic_metrics.json

python scripts/evaluate_intervention.py \
  --errors outputs/qwen3_baseline/shared_task_dev_first1000_v1_errors.jsonl \
  --intervention outputs/qwen3_interventions/v1_errors_entailment_gate.jsonl \
  --output outputs/qwen3_interventions/v1_errors_entailment_gate_metrics.json

python scripts/analyze_action_by_error.py \
  --errors outputs/qwen3_baseline/shared_task_dev_first1000_v1_errors.jsonl \
  --intervention outputs/qwen3_interventions/v1_errors_evidence_critic.jsonl \
  --output outputs/qwen3_interventions/v1_errors_evidence_critic_by_type.json

python scripts/analyze_action_by_error.py \
  --errors outputs/qwen3_baseline/shared_task_dev_first1000_v1_errors.jsonl \
  --intervention outputs/qwen3_interventions/v1_errors_entailment_gate.jsonl \
  --output outputs/qwen3_interventions/v1_errors_entailment_gate_by_type.json
```

Select an action by recovery rate and label-pair behavior, not total recovery alone. A gate that repairs NEI but destroys SUPPORTED/REFUTED must be routed selectively. Only the selected action proceeds to the 583-control regression test.
