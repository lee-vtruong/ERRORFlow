# Scripts

Place reproducible entry points here. Every experiment script must write its
configuration, commit hash, provenance fields and output paths to a Markdown
or JSON artifact under `outputs/` (which is ignored by Git).

V1 retrieval audit uses `retrieve_fts.py` with a larger candidate pool. New
artifacts include FEVER `page_id`, `sentence_id`, rank and BM25 score while
preserving the text-only `evidence` field consumed by the verifier. Evaluate
candidate recall with `evaluate_fever_retrieval.py`; see
`docs/STEP_27_V1_RETRIEVAL_AUDIT.md`.
