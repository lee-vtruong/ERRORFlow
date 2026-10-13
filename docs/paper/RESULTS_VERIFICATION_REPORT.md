# Results verification report

The numerical source of truth is `docs/reports/V3_PAPER_PACKAGE.md`, cross-checked against the specialized V3 reports and implementation.

## Verified primary results

| Dataset/evidence | B1 accuracy / F1 | A3-All accuracy / F1 | R6 accuracy / F1 | R6 route |
|---|---:|---:|---:|---:|
| AVeriTeC grouped internal | .7402 / .5460 | .7360 / .5559 | .7552 / .5577 | .1118 |
| CLIMATE-FEVER official candidates | .5212 / .3808 | .5518 / .4393 | .5375 / .4063 | .1993 |
| SciFact oracle, projected | .6067 / .5596 | .8167 / .7822 | .6367 / .6052 | .1133 |
| SciFact BM25, projected | .6100 / .5778 | .6367 / .6157 | .6267 / .6035 | .1867 |

## Statistical interpretation

- AVeriTeC grouped R6 vs B1: exact McNemar p=8.745e-5; both paired bootstrap intervals positive. This is train-selected internal evidence.
- CLIMATE-FEVER R6 vs B1: p=.00728; both intervals positive. R6 is not distinguishable from matched confidence/similarity controls.
- SciFact oracle projected R6 vs B1: p=.02246; both intervals positive. The raw four-label comparison is null and projection-sensitive.
- SciFact BM25 R6 vs B1: p=.26685; accuracy interval crosses zero, macro-F1 interval is positive. Matched-router comparisons are inconclusive.

## Non-primary results

The memory-size ablation and 357-route significance result use the historical exact-fold protocol. They are useful sensitivity analyses but are not mixed into the conservative primary AVeriTeC table.
