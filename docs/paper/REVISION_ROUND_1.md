# Paper revision round 1: method evidence and reviewer concerns

Date: 2026-10-13. Rendering intentionally deferred at the author's request.

## Completed source changes

1. Replaced the original block-diagram reference with `paper/figures/Overview-method.png`. The caption now distinguishes the offline source-memory phase from online selective correction and constrains the meaning of “internal signals” to implemented score/evidence features.
2. Added the complete matched-budget AVeriTeC table: B1, A3-All, random, confidence, margin, label-conditional, similarity-only, static-pattern, outcome-memory, and R6. No missing component-ablation value was inferred.
3. Added a direct statistical-comparison table. Exact intervals/p-values are printed only where frozen reports provide them; absent comparisons are marked NR.
4. Expanded Related Work into evidence-grounded verification, test-time self-correction, selective routing, and outcome-based memory, plus a verified conceptual comparison table.
5. Expanded utility estimation: smoothing interpretation, weight normalization, minimum support, source-label outcome definitions, zero threshold, near-boundary instability, and cost units.
6. Added an explicit B1-versus-A3 table and full immutable prompts in the appendix. The text states that A3 does not receive the B1 answer.
7. Added a transparent numerical utility calculation. It is labeled illustrative, not empirical.
8. Added `paper/scripts/export_v3_worked_example.py` to export a genuine helpful or harmful fold-safe trace from server artifacts. No synthetic claim is presented as an observed example.

## Deliberately unresolved until server export

The frozen repository does not contain the V3 prediction/memory JSONL artifacts. Therefore a genuine claim-level top-five memory example cannot be embedded locally. Run on the server:

```bash
python paper/scripts/export_v3_worked_example.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --baseline-records outputs/v3/averitec/development/baseline_scores.jsonl \
  --candidate outputs/v3/averitec/full_actions/conflicting_rescue_v1.jsonl \
  --policy outputs/v3/ablations/matched_budget_grouped090/natural_budget_predictions.jsonl \
  --memory outputs/v3/averitec/full_actions/conflicting_rescue_v1_memory.jsonl \
  --prefer helpful \
  --output outputs/v3/paper/worked_example_helpful.json
```

If the policy filename differs, locate it with:

```bash
find outputs/v3 -type f -name '*prediction*.jsonl' | grep -E 'matched|grouped090|natural'
```

The component ablations requested for claim-only, score-only, length-only, no-outcome, and no-regression R6 are new experiments. They are identified as not executed rather than assigned fabricated values.

## Checks performed without rendering

- citation-key consistency;
- Python syntax for the new trace exporter;
- Git whitespace check.

PDF compilation and page-image inspection are intentionally postponed until the next consolidated revision.
