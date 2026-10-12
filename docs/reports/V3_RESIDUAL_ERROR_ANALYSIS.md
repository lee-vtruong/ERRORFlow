# ERRORFlow V3 residual error analysis

Status: **25-EXAMPLE SMOKE COMPLETED; FULL TRAIN ANALYSIS PENDING**.

The analyzer writes complete B1 score records, residual-only JSONL, grouped fold
assignments, confusion/per-class metrics, score-margin and entropy summaries,
evidence-length associations, and explicitly unvalidated diagnosis hypotheses.
Gold/prediction transitions remain observable offline outcomes and are not used
as online causal diagnoses.

```bash
python scripts/analyze_v3_residuals.py \
  --gold data/processed/v2/averitec/train_all.jsonl \
  --predictions outputs/v3/averitec/b1_train_all.jsonl \
  --output-dir outputs/v3/averitec/development \
  --folds 5 --seed 2026
```

The command rejects the opened AVeriTeC dev holdout and labels unavailable
retrieval-score associations as missing rather than fabricating them.

## Smoke observations

The first 25 train examples contained 13 correct predictions and 12 errors.
All six CONFLICTING examples were missed, and B1 produced no CONFLICTING label.
The dominant transitions were REFUTED→SUPPORTED (5),
CONFLICTING→SUPPORTED (4), and CONFLICTING→REFUTED (2).

Errors had lower mean score margin than correct cases (0.534 versus 0.681) and
higher mean entropy (0.849 versus 0.608). Mean evidence count was similar (2.75
versus 2.85); mean evidence length was 131.4 versus 120.2 tokens. These are
small-sample hypotheses only. Causal diagnosis remains unvalidated, and no
retrieval-score association is available under oracle evidence.
