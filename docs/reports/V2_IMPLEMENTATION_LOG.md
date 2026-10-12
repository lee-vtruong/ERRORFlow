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
