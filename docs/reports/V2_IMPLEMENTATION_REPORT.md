# ERRORFlow V2 implementation report

Date: 2026-10-12

## Outcome

The repository has been migrated from a FEVER-specific experimental codebase to
an additive V2 foundation while preserving the old pipeline. The final local
regression gate at the initial V2 milestone was **50 passed, 11 warnings**; Step
later implementation steps raise it to **59 passed, 11 warnings**. The warnings are inherited Python
deprecation warnings; no test failed.

This is an implementation milestone, not an experimental result. No AVeriTeC,
SciFact, or Climate-FEVER model run was executed, and no V2 accuracy improvement
is claimed.

## Architecture now available

```text
dataset JSONL
  -> dataset adapter -> NormalizedExample
  -> frozen verifier output
  -> paired error/intervention outcomes
  -> leakage-safe ErrorMemoryRecord store
  -> compatible-memory retrieval
  -> uncertainty × expected recovery - intervention cost
  -> route or abstain
  -> unified effectiveness/safety/cost metrics
```

The key design choice is separation: dataset normalization, verifier inference,
memory construction, memory retrieval, policy selection, intervention execution,
and evaluation can each be tested independently. Legacy FEVER scripts remain in
place and are pinned by the archive.

## Implemented

- FEVER V0/V1 reference archive with commit boundary, checksum anchors,
  historical metrics, environment notes, commands, and rollback procedure.
- Canonical evidence/example contracts and adapters for FEVER, AVeriTeC,
  SciFact, and Climate-FEVER.
- Structured V2 error records with deterministic IDs, provenance, outcome,
  utility, token/call/latency cost, embedding slot, and routing eligibility.
- Idempotent JSONL memory store, lexical retrieval, split/dataset filtering,
  deterministic hashing embeddings, and lazy sentence-transformer support.
- Cost-aware memory-conditioned router with minimum support/similarity and an
  explicit abstain path.
- Policy replay over frozen predictions; interventions are injected and cannot
  silently mutate memory during evaluation.
- Accuracy, macro-F1, helpful/harmful transitions, net recovery, route rate,
  label distribution, and average LLM-call accounting.
- Atomic JSON/JSONL output, canonical config hashes, run IDs, Git/environment
  manifests, and contamination checks.
- Leakage-safe memory builder: train memory is routable; dev/test memory is
  analysis-only unless an explicitly marked unsafe override is supplied.
- Bounded AVeriTeC and SciFact smoke configurations plus server handoff steps.

## Deliberately not executed

- Dataset download: the master survey workbook was not found locally, and the
  repository has no audited V2 dataset snapshot yet.
- New model downloads: server caches should be inventoried first and disk was
  last reported at 99% utilization.
- GPU pilots or full runs: they must follow successful server tests, official
  split/checksum recording, and a bounded 25-example dry run.
- Full V1 FEVER experiment: the archive records it as incomplete rather than
  extrapolating a score.

## Acceptance boundary and remaining work

The software foundation and safety gates are ready. Research acceptance still
requires: acquire one primary dataset from an official source; validate splits;
implement dataset-specific retrieval/verifier prompts; run baseline B0; create
train-only memory; run ERRORFlow E0; execute no-memory/random-memory/no-cost
ablations; repeat seeds; bootstrap paired deltas; and consolidate result tables.

Dense FAISS indexing is an optimization after a memory-size benchmark. The
current deterministic retrieval baseline is preferable for the first smoke run
because it is auditable and requires no additional model or index download.

## Immediate handoff

Use `V2_SERVER_NEXT_STEPS.md`. Do not launch a full job immediately. First pull,
run the current test gate, inspect disk, recover the survey workbook if available,
and normalize a maximum of 25 official AVeriTeC examples.
