# ERRORFlow V3 implementation log

## Phase 0 — Preservation and audit

Status: **COMPLETED**.

- Verified V2 HEAD `6e20716b...` and historical reports.
- Created and pushed `archive/errorflow-v2-conflict-audit`.
- Froze B1 prompt SHA-256 `700ebdc1...decf7` in code and tests.
- Preserved pre-existing dirty user files without staging them.
- Added explicit rejection of opened AVeriTeC IDs `dev:25..dev:499`.

## Phase 1 — Strong verifier reproduction

Status: **full 3,068-example B1 train cache complete**.

- Added bounded, resumable, hash-validated V3 action runner.
- Added manifest and per-sample cache-key provenance.
- Reused the exact four-label Qwen scorer and immutable B1 prompt.
- Verified 25/25 cached outputs, 100 forward passes, prompt/input/config hashes, and safe bounded execution.
- Completed 3,068/3,068 B1 predictions in 364.65 seconds and 12,272 forward passes.

## Phase 2 — Residual analysis

Status: **full train residual analysis complete**.

- Added confusion/per-class metrics, score margin/entropy, evidence-size
  associations, observable transitions, and unvalidated taxonomy hypotheses.
- Added deterministic grouped folds and train/holdout contamination checks.
- Measured 797 residual errors, including 185/195 missed CONFLICTING examples.

## Phases 3–5 — Actions, memory, and router

Status: **FUNCTIONALLY IMPLEMENTED; experimental evidence pending**.

- Added A1–A5 prompt portfolio without changing B1.
- Added a deterministic train-only action-screen builder. It stratifies by
  observed gold→B1 transitions and correct-control labels, then spans each
  group's B1 margin range instead of taking a convenience or random slice.
- Full-train screen construction and A1–A5 outcome collection are complete:
  150 predictions for each of five actions, 750 total.
- Utility screening found only `conflicting_rescue_v1` globally positive after
  population-group weighting (weighted net +117.8; projected accuracy +0.03840;
  projected macro-F1 +0.01731). This is descriptive train-side screening, not a
  holdout estimate.
- Memory construction now explicitly includes correct controls so harmful
  regressions contribute negative utility. R0–R6 evaluation remains pending.
- The first full-train action cache is complete: `conflicting_rescue_v1`
  produced 3,068/3,068 predictions in 366.998 seconds and 12,272 forward
  passes. Full-train cached evaluation is the next step; A1, A2, A4, and A5
  remain screen-only.
- Full-train fold-excluded routing now supports the memory hypothesis for A3:
  R6 routed 11.64%, netted +58 corrections, and improved accuracy/macro-F1 by
  +0.01891/+0.01546 at 4.465 average passes. Always-on A3 reduced accuracy.
- Added per-example fixed-R6 materialization and paired bootstrap/exact McNemar
  tooling. Statistical execution remains pending.
- Added versioned/deactivatable residual action memory and fold-safe retrieval.
- Added R0–R6 cached-action evaluation at population-weighted matched budgets.
- Nested inner-fold selection, Architecture B, and cross-domain validation remain
  explicitly not run/not implemented.
