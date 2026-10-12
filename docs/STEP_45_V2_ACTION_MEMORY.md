# Step 45 — Train-only action-outcome memory

## Objective

Persist all three intervention outcomes for all 155 screening examples. Correct
controls are retained as negative-utility memories; dropping them would make the
router learn recovery without learning regression risk.

Each record stores observable baseline prediction and label-score profile,
claim text, action, candidate outcome, latency, forward-pass cost, and offline
train utility (+1 recovered, -1 regressed, 0 unchanged/still wrong).

Gold transition annotations may be analyzed offline but are forbidden as online
query features. Retrieval first requires the same baseline prediction, then
combines claim lexical similarity and label-score-distribution similarity.

## Expected artifact

- 155 unique train examples.
- Three actions per example.
- 465 memory records.
- Recovered outcomes: 92 total across actions.
- Regressed outcomes: 38 total across actions.

## Result

Completed and validated:

- 465 unique memory records from 155 unique train claims.
- 155 records for each of the three actions.
- Utility distribution: 92 positive, 38 negative, and 335 zero.
- Roles: 345 error records and 120 correct-control records.
- No non-train claim ID is present.

Artifact: `outputs/v2/averitec/memory/action_outcomes_train.jsonl`.
