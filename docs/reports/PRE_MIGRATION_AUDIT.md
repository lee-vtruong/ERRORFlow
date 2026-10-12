# ERRORFlow V2 pre-migration audit

Date: 2026-10-12 (Asia/Saigon)

## Decision

The repository is suitable for an incremental V2 migration, but not for a
destructive rewrite. FEVER V0/V1 is functional and historically valuable; its
large data/model artifacts live only on the server, while the local Git tree
contains uncommitted user documentation. The migration therefore uses a
reference archive, compatibility tests, and new versioned modules.

## Repository state

- Remote: `https://github.com/lee-vtruong/ERRORFlow`
- Branch: `main`
- HEAD: `a4280b8ac550909beb57b5509e10bf71cd289304`
- Tag at audit time: none
- Legacy test gate: `32 passed, 11 warnings`
- Existing architecture: FEVER preparation/retrieval, frozen Qwen verifier,
  error extraction, intervention, selective routing, evaluation, and bootstrap.

The working tree already contained modified reports and untracked research
documents. These files belong to the user. The audit did not stage, overwrite,
move, delete, stash, reset, or commit them.

## Storage and compute

The local machine has a 6 GB RTX 3050 and about 35 GB free disk, which is
appropriate for tests and small pilots, not the server-scale pipeline. The
server has an RTX 5090 and the required Qwen/BGE caches, but its filesystem was
last reported at 99% utilization with about 9.8 GB free. V2 must therefore use
streaming files, resumable jobs, compact indices, explicit cache paths, and no
duplicate copies of FEVER or model weights.

## Reusable V1 components

- JSONL contracts and deterministic claim IDs.
- FEVER label normalization and evidence resolution.
- Resumable Qwen inference.
- Retrieval provenance and retrieval recall evaluation.
- Error extraction, regression accounting, cost metrics, and paired bootstrap.

## Gaps V2 must close

- Dataset adapters are FEVER-specific; AVeriTeC and SciFact are absent.
- Error memory lacks the structured schema, embedding/index layer, lifecycle,
  deduplication, and contamination safeguards required by the master task.
- Routing is based on a narrow label/confidence rule rather than retrieved
  historical utility, uncertainty, compatibility, and explicit cost.
- There is no unified config-hashed experiment runner or consolidated result
  table across datasets/seeds/ablations.
- Current confidence is strongly overconfident on many errors and must not be
  treated as a calibrated probability.

## Missing survey workbook

`Text_Only_Fact_Checking_Training_Free_Dataset_Survey.xlsx` was searched for
under `C:/Users/ASUS` and was not found locally. Dataset/SOTA claims that depend
on that workbook must remain pending until the file is supplied or recovered.

## Migration boundary

FEVER history is frozen under `legacy/fever_v1/`. V2 development should add new
interfaces and adapters while retaining legacy script compatibility. Large
experiments are not authorized by this audit; only unit tests and bounded pilots
should run until disk headroom and dataset/model paths are confirmed.
