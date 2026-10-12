# Step 41 — AVeriTeC oracle-evidence holdout result

## Integrity

- Calibration examples excluded: `dev:0`–`dev:24`.
- Holdout evaluated: `dev:25`–`dev:499`.
- Gold/predictions/paired: 475/475/475.
- Missing/extra predictions: 0/0.
- Model forward passes: 4.0/example.
- Generated tokens: 0.0/example.

## Result

- Accuracy: **0.68632** (326/475).
- Macro-F1: **0.52417**.
- Average latency: **114.69 ms/example**.

Gold support and correct count:

- CONFLICTING: 5/35 correct.
- NOT ENOUGH INFO: 23/33 correct.
- REFUTED: 211/288 correct.
- SUPPORTED: 87/119 correct.

Largest error transition: REFUTED → NOT ENOUGH INFO (61). CONFLICTING remains
the weakest class, with only 5 correct examples. These observations document
the frozen baseline; they must not be converted into holdout-specific route
rules or thresholds.

## Scientific boundary

This is an oracle-evidence verifier result. It is not comparable to AVeriTeC
end-to-end scores because the model receives annotated question-answer evidence.
The 475 labels and errors are now sealed from method development. ERRORFlow
memory, interventions, and thresholds must be built only from official train.
