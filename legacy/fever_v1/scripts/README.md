# Original FEVER script map

The executable files remain under `scripts/` and are frozen by the audited
commit plus `manifests/checksums.sha256`. They are not copied here because a
copy would create two editable sources of truth.

| Stage | Original entry point |
|---|---|
| FEVER audit | `scripts/audit_fever.py` |
| Claim preparation | `scripts/prepare_fever_claims.py`, `scripts/prepare_fever_subset.py` |
| SQLite FTS index | `scripts/build_wiki_fts.py` |
| Lexical retrieval | `scripts/retrieve_fts.py` |
| Title retrieval | `scripts/retrieve_fever_titles.py` |
| Candidate merge | `scripts/merge_retrieval_candidates.py` |
| Cross-encoder rerank | `scripts/rerank_cross_encoder.py` |
| Qwen verifier | `scripts/run_qwen3_baseline.py` |
| Error extraction/diagnosis | `scripts/build_error_list.py`, `scripts/diagnose_errors.py` |
| Route construction/merge | `scripts/build_route_subset.py`, `scripts/merge_route_predictions.py` |
| Evaluation | `scripts/evaluate_predictions.py`, `scripts/evaluate_regression.py` |
| Significance | `scripts/paired_bootstrap.py` |

V2 code may import reusable logic from `src/errorflow`, but compatibility with
these entry points must be checked before any old command is re-run.
