# ERRORFlow V3 paper-ready result package

Status: **DRAFT RESULTS PACKAGE — EXPERIMENTS FROZEN THROUGH WP4**.

## Recommended framing

ERRORFlow V3 is a training-free, post-hoc residual correction framework for
fact verification. A frozen base verifier first predicts a verdict. A second
frozen action prompt is available as a counterfactual correction. A
non-parametric source error memory estimates whether the action is likely to
help, and a utility router invokes it selectively. No model weights are
updated.

The defensible contribution is **selective residual correction under a frozen
inference budget**, not state-of-the-art fact checking and not reliable direct
classification of conflicting evidence. Source memory uses labeled AVeriTeC
outcomes offline, so the method is training-free with respect to model
parameters but is not label-free or supervision-free.

Suggested title:

> ERRORFlow: Training-Free Selective Residual Correction for Fact Verification

## Frozen method

1. B1 scores all task labels using mean candidate-label log probability from
   Qwen3-4B-Instruct-2507.
2. A3 (`conflicting_rescue_v1`) is a second fixed prompt over the same claim and
   evidence.
3. Residual memory stores AVeriTeC train action outcomes and observable B1
   features. It does not update model parameters.
4. R6 retrieves compatible source memories and routes only when estimated net
   utility after a fixed action-cost penalty is positive.
5. Matched-route confidence and similarity routers control for extra compute.

With one action and one constant cost, `R-Memory` and R6 induce identical
rankings at a fixed route budget. The current experiments validate residual
outcome-memory routing and explicit compute accounting, not an independent
benefit from cost-aware ranking.

Frozen R6 parameters are `top_k=5`, `min_support=2`, `cost_weight=0.1`, and
utility threshold `>0`. All external target labels are read only for final
evaluation.

## Main results

| Evaluation | System | Accuracy | Macro-F1 | Route rate | Avg. forward passes | Interpretation |
|---|---|---:|---:|---:|---:|---|
| AVeriTeC train, grouped fold-excluded internal | B1 | 0.7402 | 0.5460 | 0.000 | 4.000 | Train-selected source baseline |
| AVeriTeC train, grouped fold-excluded internal | R6 | 0.7552 | 0.5577 | 0.112 | 4.447 | Conservative near-duplicate grouping; not independent holdout |
| CLIMATE-FEVER official candidates | B1 | 0.5212 | 0.3808 | 0.000 | 4.000 | Locked four-label external baseline |
| CLIMATE-FEVER official candidates | A3-All | 0.5518 | 0.4393 | 1.000 | 8.000 | Significant +3.06 accuracy, +5.85 F1 |
| CLIMATE-FEVER official candidates | R6 | 0.5375 | 0.4063 | 0.199 | 4.797 | Significant versus B1; not above confidence control |
| SciFact oracle rationale, projected | B1 | 0.6067 | 0.5596 | 0.000 | 4.000 | Projection changes 5 predictions |
| SciFact oracle rationale, projected | A3-All | 0.8167 | 0.7822 | 1.000 | 8.000 | Large adapter-mediated result; 96 projections |
| SciFact oracle rationale, projected | R6 | 0.6367 | 0.6052 | 0.113 | 4.453 | Significant, but adapter-dependent |
| SciFact open BM25 evidence, projected | B1 | 0.6100 | 0.5778 | 0.000 | 4.000 | Realistic retrieved-evidence baseline |
| SciFact open BM25 evidence, projected | A3-All | 0.6367 | 0.6157 | 1.000 | 8.000 | Directional, not significant at n=300 |
| SciFact open BM25 evidence, projected | R6 | 0.6267 | 0.6035 | 0.187 | 4.747 | Positive F1 interval; accuracy interval crosses zero |

## Statistical findings

- AVeriTeC grouped internal R6: +0.01499 accuracy and +0.01169 macro-F1; exact
  McNemar `p=8.75e-5`; both bootstrap intervals positive. The historical
  exact-duplicate-fold estimate is +0.01890/+0.01546 and is retained only as a
  sensitivity result. Both estimates remain train-selected.
- CLIMATE-FEVER A3-All versus B1: exact McNemar `p=0.00140`; accuracy delta CI
  `[0.0124, 0.0489]`; macro-F1 delta CI `[0.0355, 0.0825]`.
- CLIMATE-FEVER R6 versus B1: exact McNemar `p=0.00728`; accuracy delta CI
  `[0.0052, 0.0280]`; macro-F1 delta CI `[0.0070, 0.0451]`.
- CLIMATE-FEVER R6 is not significantly different from matched-budget
  confidence or similarity routing.
- SciFact oracle projected R6 versus B1: exact McNemar `p=0.02246`; both
  bootstrap intervals positive. The raw four-label comparison is null.
- SciFact retrieved R6 versus B1: accuracy CI `[-0.0067, 0.0400]`; macro-F1 CI
  `[0.0020, 0.0530]`. Accuracy is inconclusive; macro-F1 is positive under the
  paired bootstrap.

## Retrieval experiment

The SciFact lexical retriever searches all 5,183 official corpus documents,
retains five BM25 documents, and then five BM25 sentences. It never uses cited
document IDs, gold rationale indices, or labels for ranking. On 188 verifiable
claims, top-5 any-rationale recall is 0.6436 and complete-rationale recall is
0.6277.

Retrieval stratification shows:

| Group | N | B1 accuracy | A3 net vs B1 | R6 net vs B1 | R6 route rate |
|---|---:|---:|---:|---:|---:|
| NEI, no gold rationale | 112 | 0.4286 | +15 | +5 | 0.250 |
| Verifiable retrieval hit | 121 | 0.7851 | -1 | 0 | 0.107 |
| Verifiable retrieval miss | 67 | 0.5970 | -6 | 0 | 0.224 |

R6 does not repair absent evidence. Its useful behavior is harm avoidance: it
retains B1 accuracy on retrieval misses while always-on A3 and confidence
routing regress.

## Claim-evidence matrix

| Candidate claim | Status | Evidence boundary |
|---|---|---|
| A frozen second-pass action can improve fact verification without weight updates | Supported | Significant on CLIMATE-FEVER; directional on retrieved SciFact |
| Selective intervention reduces compute relative to always-on correction | Supported | 4.45–4.80 versus 8.0 average forward passes |
| Source-memory R6 improves its frozen B1 | Supported with scope | Significant on CLIMATE-FEVER and internal AVeriTeC; retrieved SciFact F1 positive but accuracy inconclusive |
| Error memory is better than simple matched-budget routing | Not supported externally | No significant R6 advantage over confidence/similarity controls |
| ERRORFlow reliably detects conflicting evidence | Not supported | CONFLICTING F1 remains low; gains concentrate on NEI and polarity errors |
| The method transfers without ontology adaptation | Not supported on SciFact | Raw SciFact comparison is null; oracle gain is projection-mediated |
| R6 is robust to imperfect retrieval | Partially supported | B1/R6 remain stable from oracle to BM25; n=300 and no stronger retriever comparison |
| ERRORFlow is state of the art | Not evaluated | Protocols and evidence settings are not directly comparable to leaderboard systems |

## Required limitations

1. The base model is a single Qwen3-4B checkpoint; model-family generality is
   unknown.
2. AVeriTeC source results are train-selected and not an untouched holdout.
3. The error memory is built from labeled source outcomes; “training-free”
   means no parameter optimization, not no supervision.
4. SciFact oracle results depend strongly on the frozen label projection.
5. CLIMATE-FEVER uses official retrieved-and-annotated candidates, not
   ERRORFlow retrieval.
6. SciFact open retrieval is a lexical baseline with only 64.36% top-5
   any-rationale recall.
7. External samples are modest, and several matched-router comparisons are
   statistically inconclusive.
8. CONFLICTING-class performance is weak across the four-label datasets.
9. Constant action cost makes R-Memory and R6 equivalent at a fixed budget;
   heterogeneous-cost routing has not been demonstrated.

## Recommended paper tables and figures

- Table 1: dataset, evidence condition, labels, model, and leakage boundary.
- Table 2: B1, A3-All, matched controls, and R6 quality/cost results.
- Table 3: paired significance and confidence intervals.
- Table 4: per-class F1, emphasizing NEI and CONFLICTING.
- Figure 1: B1 -> memory router -> optional A3 pipeline.
- Figure 2: accuracy and macro-F1 versus average forward passes.
- Figure 3: helpful/harmful changes by evidence condition.
- Figure 4: retrieval-hit/miss/NEI stratification.

Existing matched-budget PDFs under `outputs/v3/figures/matched_budget/` can be
used for the internal ablation section after checking labels and captions.
The reproducible external plotting entrypoint is
`scripts/plot_v3_external_results.py`; it generates five PDFs directly from
the frozen CLIMATE-FEVER, SciFact retrieved, and retrieval-stratification JSON
artifacts.

## Suggested paper structure

1. Introduction: inference-time residual errors and unnecessary second-pass
   cost.
2. Related work: fact verification, self-correction, test-time compute,
   retrieval-augmented verification, and non-parametric memory.
3. Method: B1, counterfactual action, residual memory, utility routing, and
   compute accounting.
4. Experimental protocol: contamination controls, evidence conditions,
   matched budgets, and significance tests.
5. Results: internal source analysis, external four-label transfer, SciFact
   adapter sensitivity, and retrieved-evidence robustness.
6. Analysis: per-class behavior, action helpfulness/harm, retrieval strata,
   memory size, and failure cases.
7. Limitations and responsible claims.
8. Conclusion.

## Next experiments, in priority order

1. Repeat with a second base-model family while keeping prompts and router
   selection frozen.
2. Add a predeclared stronger retriever/reranker and compare against the frozen
   BM25 condition without replacing it.
3. Use nested inner-fold action/router selection on a new source split.
4. Improve the action specifically for CONFLICTING detection, then evaluate it
   as a new version rather than overwriting A3.
5. Run multiple fixed seeds only where decoding or retrieval is stochastic.
