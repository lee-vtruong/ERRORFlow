# ERRORFlow V3 structured residual memory

Status: **IMPLEMENTED; FULL A3 TRAIN ARTIFACT AND SIZE ABLATION COMPLETED**.

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

## Full A3 memory-size ablation

| Memory records | Route rate | Net | Delta accuracy | Delta macro-F1 | Avg passes |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0 | 0.00000 | 0.00000 | 4.000 |
| 25 | 0.1102 | +39 | +0.01271 | -0.00055 | 4.441 |
| 50 | 0.1848 | +51 | +0.01662 | +0.01094 | 4.739 |
| 100 | 0.1340 | +63 | +0.02053 | +0.01374 | 4.536 |
| 200 | 0.1675 | +52 | +0.01695 | +0.00720 | 4.670 |
| all (3,068) | 0.1164 | +58 | +0.01890 | +0.01546 | 4.465 |

The curve is not monotonic. Fifty records are sufficient for positive deltas
on both metrics; 100 maximizes accuracy/net, while the full memory maximizes the
predeclared primary metric macro-F1 and uses fewer passes than the 50- and
200-record variants. The full memory remains the frozen source configuration;
100 is an exploratory accuracy-oriented alternative.
