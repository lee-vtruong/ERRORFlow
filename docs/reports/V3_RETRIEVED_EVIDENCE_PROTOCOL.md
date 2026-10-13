# ERRORFlow V3 retrieved-evidence validation protocol

Status: **PRIMARY RETRIEVED-EVIDENCE EVALUATION COMPLETE; PER-CLASS EXTRACTION
PENDING**.

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

## Retrieval checkpoint

The frozen retriever produced 300 unique examples and exactly five sentences
per example from 5,183 corpus documents. The corpus and claim hashes reproduce
the earlier oracle experiment. All 1,500 evidence records have source
`scifact_open_bm25`; the manifest confirms no use of cited document IDs, gold
rationales, or labels for ranking. The artifact occupies 900 KiB, leaving 9.7
GiB free on the server.

Among 188 verifiable claims, top-5 any-rationale sentence recall is 0.6436,
complete-rationale recall is 0.6277, and mean best-rationale coverage is
0.6356. At top 1 these metrics are 0.3936, 0.3777, and 0.3856; at top 3 they
are 0.5585, 0.5479, and 0.5532. This fixed, imperfect retrieval condition is
accepted without parameter tuning and will be compared directly with the
existing oracle-rationale result.

## Retrieved-evidence result

B1 reaches 61.00% accuracy and 57.78 macro-F1. A3-All reaches 63.67% accuracy
and 61.57 macro-F1, making 22 helpful and 14 harmful changes (net +8) at eight
forward passes per example. Its deltas over B1 are +2.67 accuracy points and
+3.80 macro-F1 points, but both paired-bootstrap intervals cross zero and
McNemar gives `p=0.24298`.

Frozen source-only R6 routes 56/300 examples (18.67%) and requires 4.747
forward passes per example. It makes nine helpful and four harmful changes
(net +5), reaching 62.67% accuracy and 60.35 macro-F1. The +1.67-point
accuracy delta over B1 is not significant (`p=0.26685`, bootstrap interval
`[-0.0067, 0.0400]`). The +2.58-point macro-F1 delta has a positive bootstrap
interval `[0.0020, 0.0530]`, with bootstrap probability 0.983 of a positive
delta.

At the same 56-example route count, confidence and similarity controls both
remain at 61.00% accuracy, with macro-F1 58.40 and 57.68 respectively. R6 is
directionally better than both, but the direct paired intervals cross zero.
A3-All is also not distinguishable from R6: exact McNemar `p=0.67764`, with
both accuracy and macro-F1 intervals crossing zero. R6 therefore preserves
most observed A3 quality at substantially lower compute in this 300-example
condition, but no routing-superiority claim is supported.

## Oracle-versus-retrieved diagnosis

Retrieved B1 is essentially stable relative to oracle B1: accuracy changes
from 60.67% to 61.00%, while macro-F1 rises from 55.96 to 57.78. R6 changes
from 63.67%/60.52 to 62.67%/60.35, a small loss. By contrast, A3-All falls
from the earlier projected oracle result of 81.67%/78.22 to 63.67%/61.57.

This large A3 difference is tied to the label adapter. Under oracle evidence,
A3 produced 96 CONFLICTING outputs before projection; under retrieved evidence
it produces only five. Retrieved B1, A3-All, and R6 project only five, five,
and four predictions respectively. Thus the retrieved result is not driven by
mass conversion of CONFLICTING to NEI, and it confirms that the very large
oracle A3 score was evidence-condition and adapter dependent.

The defensible WP4 conclusion is that selective R6 is comparatively robust to
imperfect lexical evidence and retains a positive macro-F1 signal over B1.
Always-on A3 remains directionally positive but loses its oracle advantage.
Accuracy significance and superiority over matched-budget controls are not
established on 300 examples.

## Retrieval-condition interaction

Post-hoc stratification uses gold rationale IDs only after all predictions and
routes are frozen. It separates 121 verifiable claims with at least one
retrieved rationale sentence, 67 verifiable retrieval misses, and 112 NEI
claims with no gold rationale.

On verifiable hits, B1 is already strong at 78.51% accuracy. A3-All produces a
net loss of one correct prediction, while R6 makes no net correctness change;
R6 routes only 10.74% of this group. On verifiable misses, B1 scores 59.70%,
A3-All loses six net correct predictions and falls to 50.75%, whereas R6 has
net zero, preserves 59.70% accuracy, and raises within-group macro-F1 from
0.4435 to 0.4625. The confidence control is harmful on misses (net -7).

The overall positive action signal comes from the 112 NEI claims. B1 accuracy
is 42.86%; A3-All gains 15 net correct predictions and reaches 56.25%. R6
routes 25.00%, gains five net correct predictions, and reaches 47.32%.
Confidence routing gains seven in this group, while similarity routing gains
one.

Across the three strata, A3's net changes sum to +8 and R6's to +5, matching
the aggregate accounting exactly. R6 routes less often on verifiable hits
(10.74%) than on retrieval misses (22.39%) or NEI (25.00%). Although the router
does not observe gold retrieval success, its frozen uncertainty and memory
features correlate with weak-evidence conditions. The result supports a
selective-safety interpretation: R6 captures some NEI benefit while avoiding
the substantial always-on A3 regression on retrieval misses. It does not show
that R6 repairs missing evidence.
