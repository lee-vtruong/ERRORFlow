# Frozen R6-NLF evaluation results

The canonical R6-NLF implementation was evaluated after Group-C development
selection. It retains the established R6 claim tokenizer (`[\w'-]+`), top-5
retrieval, minimum support 2, similarity weights `0.35/0.50/0.15`, utility
penalty 0.1, and the fixed `conflicting_rescue_v1` action. Only the hard
baseline-label compatibility filter is removed.

## Main natural-routing results

| Protocol | B1 Acc. | B1 F1 | NLF routes | NLF Acc. | NLF F1 | Helpful | Harmful | Net | Passes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AVeriTeC grouped 0.90 | .7402 | .5460 | 375 (12.22%) | .7585 | .5666 | 94 | 38 | +56 | 4.489 |
| CLIMATE-FEVER candidates | .5212 | .3808 | 292 (19.02%) | .5368 | .4117 | 50 | 26 | +24 | 4.761 |
| SciFact oracle, projected | .6067 | .5596 | 24 (8.00%) | .6300 | .5951 | 9 | 2 | +7 | 4.320 |
| SciFact BM25, projected | .6100 | .5778 | 62 (20.67%) | .6300 | .6075 | 10 | 4 | +6 | 4.827 |

## NLF change over B1

| Protocol | Delta Acc. | Acc. 95% CI | Delta F1 | F1 95% CI | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| AVeriTeC | +.01825 | [.01076, .02575] | +.02055 | [.00865, .03337] | 1.19e-6 |
| CLIMATE-FEVER | +.01564 | [.00456, .02736] | +.03091 | [.01378, .04892] | .00791 |
| SciFact oracle | +.02333 | [.00333, .04667] | +.03553 | [.01089, .06457] | .06543 |
| SciFact BM25 | +.02000 | [-.00333, .04333] | +.02967 | [.00434, .05728] | .17957 |

The bootstrap and exact paired tests answer different questions and need not
agree in small samples. SciFact oracle and BM25 accuracy are therefore not
described as confirmed gains despite positive point estimates.

## Matched NLF-budget comparisons

| Protocol | NLF vs HardLabel Delta Acc./F1 | McNemar p | NLF vs Confidence Delta Acc./F1 | McNemar p |
|---|---:|---:|---:|---:|
| AVeriTeC | +.00228 / +.00435 | .3240 | +.01793 / +.01515 | 5.29e-5 |
| CLIMATE-FEVER | +.00065 / +.00675 | 1.000 | +.00130 / -.00421 | .9196 |
| SciFact oracle | .00000 / +.00243 | 1.000 | +.00333 / +.01366 | 1.000 |
| SciFact BM25 | +.01000 / +.01032 | .2500 | +.02333 / +.02640 | .1671 |

Only the AVeriTeC NLF-versus-confidence comparison is clearly positive. The
external results support transfer over B1, not universal router superiority.

## Cross-label retrieval diagnostics

| Protocol | Cross-label neighbors | Neighbor rate | Queries with cross-label neighbor | Routed NLF queries with cross-label neighbor |
|---|---:|---:|---:|---:|
| AVeriTeC | 619 | 4.04% | 339 (11.05%) | 123 (32.80%) |
| CLIMATE-FEVER | 459 | 5.98% | 232 (15.11%) | 90 (30.82%) |
| SciFact oracle | 135 | 9.00% | 59 (19.67%) | 9 (37.50%) |
| SciFact BM25 | 77 | 5.13% | 38 (12.67%) | 22 (35.48%) |

## Runtime boundary

Fresh single-process CPU mean lookup time is 35.3--45.9 ms for NLF and
15.1--18.1 ms for HardLabel. These measurements exclude B1/A3 inference and
GPU synchronization. They support a memory-overhead statement, not an
end-to-end latency or monetary-cost claim.

## Group-C tokenizer audit

The broad Group-C sweep used `\w+` tokenization for claim Jaccard, while the
established R6 memory implementation uses `[\w'-]+`. Consequently, the sweep
reported 347 HardLabel and 386 NLF routes, while canonical replay reports 343
and 375. Main and external results use canonical replay. The sweep remains a
development ablation and is explicitly labeled rather than mixed with the
frozen result.
