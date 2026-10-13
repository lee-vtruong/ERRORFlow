# Full-paper audit

## Scope and consistency

- [x] The title and abstract say parameter-training-free, not supervision-free.
- [x] AVeriTeC internal selection is not described as independent held-out confirmation.
- [x] Historical 357-route results are excluded from the primary table.
- [x] SciFact oracle, SciFact BM25, and raw/projected label spaces are separated.
- [x] External matched-router non-significance and weak CONFLICTING performance are retained.
- [x] Compute is expressed as model forward passes; cached component latency is not presented as fresh end-to-end latency.
- [x] Literature comparisons are conceptual unless protocol compatibility is established.

## Reproducibility gaps

The paper source, verified numeric snapshot, figure generator, citation checker, and claim ledger are tracked. Raw datasets, model weights, and V3 result JSONL files are intentionally not tracked. A clean clone therefore reproduces the manuscript and figures but requires the documented server archive to recompute experimental statistics.

## Submission blockers

1. Replace anonymous author/affiliation and acknowledgment placeholders.
2. Select a venue and reconcile its page/appendix policy.
3. Export a checksum-addressed server artifact archive.
4. Add an environment lock with exact Transformers, PyTorch, PEFT, CUDA, model revision, and tokenizer revision.
5. Obtain a redistribution/legal decision for CLIMATE-FEVER; current policy is do not redistribute.

## Build and visual QA

- Build: TeX Live 2026 `latexmk`, Springer LNCS 2026 class supplied in the workspace.
- Output: 10 pages, US Letter, 254,538 bytes.
- Citation check: 21 entries; no missing, duplicate, or uncited keys.
- LaTeX check: no overfull boxes, unresolved references, or undefined citations.
- Visual check: all 10 rendered page images inspected; no clipping, overlap, blank pages, or table overflow.
- Regression check: 115 tests passed; 11 pre-existing Python deprecation warnings.
