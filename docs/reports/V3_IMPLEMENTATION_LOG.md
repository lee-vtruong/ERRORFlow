# ERRORFlow V3 implementation log

## Phase 0 — Preservation and audit

Status: **COMPLETED**.

- Verified V2 HEAD `6e20716b...` and historical reports.
- Created and pushed `archive/errorflow-v2-conflict-audit`.
- Froze B1 prompt SHA-256 `700ebdc1...decf7` in code and tests.
- Preserved pre-existing dirty user files without staging them.
- Added explicit rejection of opened AVeriTeC IDs `dev:25..dev:499`.

## Phase 1 — Strong verifier reproduction

Status: **IMPLEMENTED; 25-example server pilot pending**.

- Added bounded, resumable, hash-validated V3 action runner.
- Added manifest and per-sample cache-key provenance.
- Reused the exact four-label Qwen scorer and immutable B1 prompt.

## Phase 2 — Residual analysis

Status: **IMPLEMENTED; train result pending**.

- Added confusion/per-class metrics, score margin/entropy, evidence-size
  associations, observable transitions, and unvalidated taxonomy hypotheses.
- Added deterministic grouped folds and train/holdout contamination checks.

## Phases 3–5 — Actions, memory, and router

Status: **FUNCTIONALLY IMPLEMENTED; experimental evidence pending**.

- Added A1–A5 prompt portfolio without changing B1.
- Added versioned/deactivatable residual action memory and fold-safe retrieval.
- Added R0–R6 cached-action evaluation at population-weighted matched budgets.
- Nested inner-fold selection, Architecture B, and cross-domain validation remain
  explicitly not run/not implemented.
