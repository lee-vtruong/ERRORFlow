# Step 33 — Regression-test and tune the V1 REFUTED route

## Candidate policy

Route claims whose B1 baseline prediction is `REFUTED` to `entailment_gate`. The 1K baseline contains 382 such claims: 151 baseline errors and 231 baseline-correct controls. Running the action on all 382 allows direct recovery/regression measurement and a confidence sweep within this observable family.

## Build and run the complete route family

```bash
python scripts/build_route_subset.py \
  --claims data/processed/fever/shared_task_dev_first1000_bge_top5.jsonl \
  --baseline outputs/qwen3_baseline/shared_task_dev_first1000_v1_bge_top5.jsonl \
  --prediction REFUTED \
  --output data/processed/fever/shared_task_dev_first1000_v1_refuted_route.jsonl
```

Expected: `route_records=382`.

```bash
python scripts/run_qwen3_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input data/processed/fever/shared_task_dev_first1000_v1_refuted_route.jsonl \
  --output outputs/qwen3_interventions/v1_refuted_entailment_gate.jsonl \
  --split shared_task_dev_first1000_v1_refuted_route \
  --intervention entailment_gate
```

Use `--resume` if interrupted.

## Measure route-all recovery and regression

```bash
python scripts/evaluate_regression.py \
  --gold data/processed/fever/shared_task_dev_first1000_bge_top5.jsonl \
  --baseline outputs/qwen3_baseline/shared_task_dev_first1000_v1_bge_top5.jsonl \
  --intervention outputs/qwen3_interventions/v1_refuted_entailment_gate.jsonl \
  --output outputs/qwen3_interventions/v1_refuted_entailment_gate_regression.json
```

Expected pairing: 382 total, 151 baseline-wrong and 231 baseline-correct. Recovery should reproduce 47/151 if inference and inputs are deterministic.

## Sweep confidence within REFUTED predictions

```bash
python scripts/sweep_route_threshold.py \
  --gold data/processed/fever/shared_task_dev_first1000_bge_top5.jsonl \
  --baseline outputs/qwen3_baseline/shared_task_dev_first1000_v1_bge_top5.jsonl \
  --intervention outputs/qwen3_interventions/v1_refuted_entailment_gate.jsonl \
  --prediction REFUTED \
  --output outputs/qwen3_interventions/v1_refuted_entailment_thresholds.json
```

Select the threshold using both Accuracy and Macro-F1, with route rate/cost as a tie-breaker. Do not merge or report E1 until recovery, regression and the complete threshold table are available.

## Regression and threshold result

Route-all is rejected: it recovers `47/151` errors but regresses `65/231` correct controls, net `-18`. The confidence sweep exposes two Pareto candidates:

| Max confidence | Routed | Route rate | Accuracy | Macro-F1 |
|---:|---:|---:|---:|---:|
| baseline | 0 | 0.0% | 58.30% | 53.36% |
| 0.70 | 18 | 1.8% | 58.70% | 54.32% |
| 0.90 | 49 | 4.9% | 58.30% | 54.49% |

The frozen candidate policy is `prediction=REFUTED and confidence<=0.90`: it maximizes Macro-F1 among tested thresholds, does not reduce Accuracy, and stays within the approximately 5% extra-call budget used by ERRORFlow V0. The same numeric threshold as V0 is coincidental; it was independently selected from the B1 sweep with a different retrieval pipeline and intervention.

Proceed to selective merge and paired bootstrap using `docs/STEP_34_V1_SELECTIVE_EVALUATION.md`.
