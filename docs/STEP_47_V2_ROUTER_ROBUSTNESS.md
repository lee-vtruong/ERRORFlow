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

## Pass gate before holdout

- Accuracy and macro-F1 bootstrap intervals should preferably exclude zero.
- Frozen routing should beat all global `always_*` policies.
- Random same-budget routing should rarely match the frozen delta.
- No dev/holdout labels or intervention outcomes may be read.

If the gate passes, precompute the three intervention candidates on the sealed
475-example holdout without labels, replay this frozen router once, then perform
one final paired evaluation. Do not tune the policy after seeing that result.
