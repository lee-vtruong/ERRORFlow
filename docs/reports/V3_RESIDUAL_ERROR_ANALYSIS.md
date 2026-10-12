# ERRORFlow V3 residual error analysis

Status: **IMPLEMENTED; AVERITEC TRAIN RESULT PENDING**.

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
