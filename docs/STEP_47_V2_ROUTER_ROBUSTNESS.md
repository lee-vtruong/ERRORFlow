# Step 47 — Freeze and stress-test the V2 router

## Decision

Freeze `averitec_errorflow_router_v1` with `k=3`, minimum support 2, minimum
utility 0, cost weight 0.10, and inverse-confidence uncertainty. The policy can
observe claim text, baseline prediction, baseline score distribution, retrieved
train memories, and action cost. It cannot observe gold labels or gold error
transitions online.

## Why another train-only gate is required

The selected policy improved the population-weighted leave-one-example-out
estimate by +6.30 accuracy points and +3.26 macro-F1 points. However, it was
selected from 216 configurations evaluated on the same 155-example screen.
Before a one-shot holdout evaluation, quantify uncertainty and verify that the
gain is better than no memory, global actions, and random routing.

## Server command

Use the same population, screen, baseline, memory, and three candidate files as
Step 46:

```bash
python scripts/evaluate_v2_router_robustness.py \
  --population-gold data/processed/v2/averitec/train_all.jsonl \
  --screen-gold data/processed/v2/averitec/train_intervention_screen.jsonl \
  --baseline outputs/v2/averitec/oracle_train_all_s0.jsonl \
  --memory outputs/v2/averitec/memory/action_outcomes_train.jsonl \
  --candidate \
    conflict_audit_v1=outputs/v2/averitec/intervention_screen/conflict_audit_v1.jsonl \
    coverage_audit_v1=outputs/v2/averitec/intervention_screen/coverage_audit_v1.jsonl \
    refutation_audit_v1=outputs/v2/averitec/intervention_screen/refutation_audit_v1.jsonl \
  --config configs/v2/averitec_router_v1.json \
  --bootstrap-samples 5000 \
  --random-trials 500 \
  --seed 2026 \
  --output outputs/v2/averitec/memory/router_v1_robustness.json \
  | tee outputs/logs/averitec_router_v1_robustness.log
```

If the actual Step 46 screen or candidate filenames differ, reuse those exact
paths; do not regenerate inference outputs.

## Result

- Accuracy delta is +0.06304 with CI95% `[0.00713, 0.10939]` and
  `P(delta > 0) = 0.9858`.
- Macro-F1 delta is +0.03258, but CI95% `[-0.01740, 0.07575]` includes zero.
- Only 0.2% of random same-budget trials matched the frozen accuracy delta.
- The global conflict audit achieved higher raw train estimates: +0.07562
  accuracy and +0.05575 macro-F1, but routed 100% rather than 29.76%.
- At the frozen cost weight 0.10, accuracy utility is 0.71972 for ERRORFlow,
  0.68644 for baseline, and 0.66206 for always-conflict.
- No holdout label was read by this robustness run.

The gate therefore passes for **cost-aware selective routing**, but not for a
claim of best raw quality. The final holdout must report baseline, ERRORFlow,
and always-conflict together so memory conditioning is separated from the
benefit of simply applying the strongest prompt globally.

If the gate passes, precompute the three intervention candidates on the sealed
475-example holdout without labels, replay this frozen router once, then perform
one final paired evaluation. Do not tune the policy after seeing that result.
