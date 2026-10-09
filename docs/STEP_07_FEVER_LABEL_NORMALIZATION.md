# Step 07 — FEVER label normalization

## Result of audit

The downloaded files contain FEVER labels:

```text
SUPPORTS
REFUTES
NOT ENOUGH INFO
```

ERRORFlow uses the verifier-facing vocabulary:

```text
SUPPORTED
REFUTED
NOT ENOUGH INFO
```

The mapping is explicit in `src/errorflow/datasets.py`; unknown labels fail closed.

## Observed distribution

- Train: 145,449 records; SUPPORTS 80,035, REFUTES 29,775, NEI 35,639.
- Shared-task dev: 19,998 records, perfectly balanced.
- Paper dev: 9,999 records, perfectly balanced.

The train split is imbalanced, so future reports must include Macro-F1 and per-class metrics, not accuracy alone.

## Evidence constraint

The current audit only confirms labels and row counts. It does not establish that passage text is available to the verifier. Evidence references must be resolved against a Wikipedia corpus before the main experiment.
