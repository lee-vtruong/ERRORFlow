# ERRORFlow V3 residual error analysis

Status: **FULL TRAIN ANALYSIS COMPLETED**.

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

## Full train observations

- Accuracy 0.74022; macro-F1 0.54602; 797/3,068 errors.
- CONFLICTING remains the critical weakness: recall 0.05128, F1 0.08230, and
  185/195 examples missed.
- Largest transitions: REFUTED→SUPPORTED 239, REFUTED→NEI 109,
  CONFLICTING→REFUTED 90, CONFLICTING→SUPPORTED 90, and SUPPORTED→REFUTED 90.
- Correct/error mean margins are 0.565/0.331; mean entropies are 0.744/1.011.
  This supports confidence routing as a serious comparator but does not prove
  calibration.
- Error cases have longer evidence (125.7 versus 104.3 mean tokens) and more
  evidence items (3.23 versus 2.60). These are associations, not causes.

The next action screen spans the margin range inside every observable train
transition and correct-control label. This prevents a screen containing only
easy low-confidence errors.
