# ERRORFlow V3 retrieved-evidence validation protocol

Status: **PROTOCOL FROZEN; RETRIEVAL ARTIFACT PENDING**.

## Objective

WP4 measures whether the frozen ERRORFlow V3 verifier, intervention, and
source-memory router remain useful when oracle rationales are replaced with
evidence retrieved by ERRORFlow. It reuses the 300 official SciFact dev claims
from the completed cross-domain experiment so that the evidence condition is
the only intended change.

## Retrieval protocol

The retriever searches the complete official SciFact corpus. It first ranks
documents with dependency-free BM25 over title plus abstract, retains five
documents, then ranks their individual abstract sentences with BM25 and
retains five sentences. Parameters are frozen at `k1=1.2`, `b=0.75`,
`top_documents=5`, and `top_sentences=5` before retrieval results are read.

Retrieval must not inspect `cited_doc_ids`, gold rationale sentence indices,
gold rationale labels, or claim labels. Those fields are read only by the
post-retrieval evaluator. This is an open-corpus lexical retrieval condition,
not a document-oracle or sentence-oracle condition.

## Frozen downstream systems

The comparison retains the same Qwen3-4B-Instruct-2507 model, B1
`conflict_audit_v1`, A3 `conflicting_rescue_v1`, complete AVeriTeC train-only
memory, R6 parameters, seed, four-label scoring, and predeclared SciFact
projection (`CONFLICTING -> NOT ENOUGH INFO`) used in the oracle-evidence
experiment. Confidence and similarity controls are matched to the natural R6
route count without target labels.

No retrieval parameter, prompt, router threshold, memory record, or label
mapping may be changed after viewing SciFact retrieval or verification
metrics. Any later reranker is a separately versioned experiment and cannot
replace this lexical baseline.

## Evaluation

Retrieval is evaluated at sentence ranks 1, 3, and 5 using claim-level
any-rationale recall, complete-rationale recall, and mean best-rationale
coverage on verifiable claims. Verification reports projected three-label
accuracy, macro-F1, per-class F1, helpful/harmful changes, route rate, model
forward passes, exact McNemar, and paired bootstrap intervals.

Results are compared with the already frozen SciFact oracle-rationale
condition. This comparison diagnoses retrieval loss; it is not an official
SciFact end-to-end leaderboard score because the verifier and routing method
were developed on AVeriTeC and the evaluation uses the public SciFact dev set.

## Resource boundary

The corpus already exists on the server. The implementation creates only a
300-row retrieved JSONL artifact and small metrics/prediction files. No model,
embedding index, or external knowledge store is downloaded, which is required
while the server has approximately 9.7 GiB free.
