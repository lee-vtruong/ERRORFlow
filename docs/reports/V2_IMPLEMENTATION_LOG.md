# ERRORFlow V2 implementation log

## Step 0 — Pre-migration audit and FEVER archive

Status: completed locally on 2026-10-12.

- Recorded repository, environment, storage, models, datasets, results, and
  missing survey workbook.
- Preserved FEVER V0/V1 by commit reference, SHA-256 anchors, frozen configs,
  command ledger, historical metrics, and rollback instructions.
- Kept all pre-existing modified/untracked files untouched.
- Verification: 32 legacy tests passed before V2 edits.

## Step 1 — V2 contracts and safe I/O

Status: implemented and covered by the local regression suite.

- Added canonical dataset/evidence contracts and a versioned structured
  `ErrorMemoryRecord` while retaining the legacy `ErrorMemoryEntry`.
- Added deterministic config hashes, atomic JSON/JSONL writes, strict JSONL
  parsing, and completed-ID discovery for resumable jobs.

## Step 2 — Dataset adapters

Status: implemented for schema normalization; dataset downloads are not run.

- Added adapters for FEVER, AVeriTeC, SciFact, and Climate-FEVER.
- Adapters normalize IDs, labels, claim text, evidence text, provenance, and
  dataset-specific metadata into one contract.

## Step 3 — Error memory and routing foundation

Status: implemented as a dependency-free reference baseline.

- Added deterministic memory IDs, idempotent upsert, JSONL persistence,
  lexical similarity retrieval, dataset/split filters, and routing eligibility.
- Added a cost-aware router that combines neighbor recovery, query uncertainty,
  support, similarity, and expected intervention cost, with explicit abstention.
- Lexical retrieval is intentionally the transparent baseline. Dense embeddings
  remain available through a lazy sentence-transformer backend; deterministic
  hashing embeddings support offline tests. A production FAISS index remains a
  subsequent optimization, not a claimed result.

## Step 4 — Evaluation and CLI foundation

Status: implemented; no new model experiment has been run.

- Added unified accuracy, macro-F1, helpful/harmful, net recovery, route rate,
  and average LLM-call accounting.
- Added `errorflow normalize` and `errorflow inspect-memory` CLI commands.
- Added bounded dry-run configs for AVeriTeC and SciFact.

## Step 5 — Reproducible policy replay

Status: implemented and covered by unit tests.

- Added stable run IDs derived from canonical config hashes plus dataset, split,
  and seed; manifests record Git state, Python/platform, inputs, and outputs.
- Added a hard contamination guard forbidding dev/validation/test memory during
  evaluation and a policy replay layer that never mutates memory online.
- Added lazy dense-embedding support and a deterministic offline fallback.
- Intervention implementations remain injected callables so the policy layer is
  testable independently from Qwen or remote retrieval.

## Research status

This implementation work does not establish an effectiveness gain. The only
reported experimental values remain the archived FEVER results. V2 requires
dataset acquisition, split validation, bounded pilots, ablations, bootstrap
confidence intervals, and then controlled full runs.

## Step 6 — Leakage-safe memory construction

Status: implemented and unit tested.

- Added a builder for paired baseline/intervention outcomes with deterministic IDs, recovery/cost fields, and source-run provenance.
- Baseline-correct examples are excluded from error memory.
- Records originating from dev/validation/test are non-routable by default. An explicit unsafe override exists only for diagnostics and is marked in each record's metadata.

## Step 7 — Official AVeriTeC acquisition path

Status: implemented locally; server download pending.

- Selected the authors' official repository and restricted acquisition to the small train/dev annotation files under CC BY-NC 4.0.
- Added atomic download, a 100 MiB safety ceiling, schema audit, row/label counts, SHA-256 checksums, and a persisted source manifest.
- Corrected the adapter for official JSON-array files and `source_url` fields.
- Added `--limit` so the first server normalization is capped at 25 examples.

## Step 8 — Stable IDs and four-label verifier smoke

Status: implemented locally; server rerun pending.

- Fixed missing official AVeriTeC IDs using deterministic `split:index` IDs.
- Added empty/duplicate-ID rejection during normalization and safe inference resume.
- Added a four-label AVeriTeC prompt/parser without changing FEVER's default three-label behavior.
- Added a canonical V2 Qwen runner and strict ID-paired evaluator.
- Classified the upcoming 25-example run as oracle-evidence diagnostics, not end-to-end retrieval evaluation.

## Step 9 — Oracle smoke result and generation audit

Status: smoke completed; diagnostic pending server execution.

- Integrity: 25/25 paired, no missing or extra predictions.
- Result: accuracy 0.64, macro-F1 0.49485.
- No CONFLICTING prediction was produced; REFUTED had six errors.
- Average generated tokens equaled the 32-token cap, so scaling is paused until raw-output truncation is measured.
- Added a diagnostic artifact containing transition counts, generation-limit rate, confidence provenance, and error raw outputs.

## Step 10 — Four-label decoding ablation

Status: implemented locally; server comparison pending.

- Confirmed 25/25 outputs hit the cap, but labels preceded truncated explanations.
- Added an explicit AVeriTeC four-label definition profile.
- Added short-generation controls and mean conditional label log-probability scoring.
- Label scoring records all candidate scores, candidate token lengths, four forward passes, and confidence provenance.

## Step 11 — Protocol selection and holdout freeze

Status: S0 selected; 475-example server run pending.

- S0 matched G0 accuracy (0.64), slightly improved macro-F1 (0.49762 vs 0.49485), and reduced latency by about 5.2x.
- G1/S1 reduced accuracy to 0.60; explicit definitions increased REFUTED-to-NEI errors and were rejected.
- Froze the first 25 dev examples as calibration and excluded them from the 475-example holdout.
- Added offset normalization and explicit average model-forward-pass accounting.

## Step 12 — Holdout result and train-only transition

Status: holdout baseline completed; train baseline pending.

- Holdout S0: accuracy 0.68632, macro-F1 0.52417, 114.69 ms/example.
- Integrity: 475/475 paired, zero missing/extra, four forward passes/example.
- Sealed holdout labels/errors from further method and threshold development.
- Added per-class precision/recall/F1 reporting for future evaluations.
- Next memory source is exclusively the 3,068-example official train split.

## Step 13 — Train baseline and intervention screen

Status: train baseline completed; candidate screen pending.

- Train S0: 3,068/3,068 paired, accuracy 0.68644, macro-F1 0.50691.
- Identified 962 train errors without reading sealed holdout errors for method development.
- Added deterministic transition-stratified error sampling plus per-label correct controls.
- Added three auditable intervention profiles and evaluation of recovery, regression, net change, latency, and forward-pass cost.

## Step 14 — Aggregate intervention result

Status: aggregate complete; action matrix pending.

- Conflict audit: 34 helpful, 12 harmful, net +22.
- Coverage audit: 30 helpful, 14 harmful, net +16.
- Refutation audit: 28 helpful, 12 harmful, net +16.
- Rejected selecting a global winner from the error-heavy stratified aggregate.
- Added per-group utility comparison while explicitly forbidding gold transition names as online route features.

## Step 15 — Action-outcome memory foundation

Status: implemented locally; server artifact pending.

- Matrix confirmed that action utility varies sharply by error transition and controls.
- Added memory construction for all action outcomes, retaining both recovered errors and regressed controls.
- Added prediction-compatible retrieval using baseline label plus label-score-distribution and lexical similarity.
- Updated routing to optimize signed empirical utility rather than recovery alone.

## Step 16 — Population-weighted router cross-validation

Status: server sweep complete; first-ranked policy frozen.

- Added leave-one-example-out retrieval so a query cannot reuse its own action outcomes.
- Added action-balanced top-k retrieval and exact baseline-prediction compatibility.
- Reweighted the error-heavy screen to the true 3,068-example train transition distribution.
- Added uncertainty, support, utility, cost, and neighborhood-size sweep without reading holdout outcomes.
- Best weighted train estimate: +0.06304 accuracy, +0.03258 macro-F1, and 0.29762 route rate.

## Step 17 — Router freeze and robustness gate

Status: implemented locally; server bootstrap/ablation run pending.

- Recorded the stable top-ten neighborhood and froze the first-ranked router before holdout intervention inference.
- Added transition-stratified population-weighted bootstrap intervals.
- Added no-memory, always-action, and random same-budget negative controls.
- Preserved the train-only boundary; the robustness evaluator never opens holdout labels.

## Step 18 — One-shot holdout runner

Status: three 475-example holdout candidate runs complete; frozen replay and evaluation pending.

- Added exact replay of the frozen action-balanced router over precomputed candidates.
- Added canonical four-label paired bootstrap evaluation.
- Fixed baseline, ERRORFlow, and always-conflict as the reported holdout comparison before candidate outcomes are observed.
