# ERRORFlow V3 structured residual memory

Status: **IMPLEMENTED; TRAIN ARTIFACT PENDING**.

The memory stores one versioned action outcome per train example/action. It
records recovery, regression, cost, provenance, observable applicability
features, offline-only labels, validation status, and activation state.

Implemented operations:

- deterministic IDs and deduplicating upsert;
- rejection of non-train and opened holdout records;
- lexical, label-score-profile, and evidence-length similarity;
- exact B1-prediction compatibility;
- action filtering and memory-size limits `0/25/50/100/200`;
- fold exclusion during retrieval;
- version checks and explicit deactivation with reason;
- smoothed utility helper for low-support aggregate estimates.

Gold labels, correctness, and error transitions are retained only as offline
outcome provenance. Retrieval never consumes them.
