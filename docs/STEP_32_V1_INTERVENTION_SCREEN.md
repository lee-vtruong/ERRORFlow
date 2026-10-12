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

## Screening result

Evidence critic recovered `37/417` errors (`8.87%`), while entailment gate recovered `59/417` (`14.15%`). The useful observable family is baseline prediction `REFUTED`:

| Baseline prediction | Evidence critic | Entailment gate |
|---|---:|---:|
| REFUTED | 31/151 (20.53%) | 47/151 (31.13%) |
| SUPPORTED | 2/223 (0.90%) | 11/223 (4.93%) |
| NOT ENOUGH INFO | 4/43 (9.30%) | 1/43 (2.33%) |

The entailment gate is selected only for the `REFUTED` family. It is not justified for SUPPORTED or NEI predictions. The next step runs it on all 382 baseline-REFUTED claims, including 231 correct controls, and sweeps confidence only within this family. See `docs/STEP_33_V1_ROUTE_REGRESSION.md`.
