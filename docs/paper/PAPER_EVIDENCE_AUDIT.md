# ERRORFlow V3 paper evidence audit

Date: 2026-10-13. Scope: implementation semantics, tracked reports, and the paper package.

## Verdict

The headline counts and point estimates are internally consistent. Exact McNemar p-values were independently recomputed from discordant counts. Code inspection confirms the label-scoring, memory retrieval, fold exclusion, routing, and cost formulas described in the manuscript.

The checkout does not contain the large V3 JSON/JSONL outputs because `outputs/` is ignored. Consequently, bootstrap intervals, conservative per-class values, and Holm adjustments are supported by tracked reports and producer code, but cannot be regenerated locally without the server artifacts. The manuscript labels this limitation rather than treating Markdown as raw evidence.

## Protocol-critical findings

- B1 and A3 are frozen prompts over the same model; A3 does not consume B1 output or generate a rationale under label scoring.
- Four-label scoring uses four independent forward passes and averages conditional token log-probabilities.
- Confidence is softmax over mean label scores, not calibrated sequence probability.
- R6 uses source labels only through offline action outcomes. It is parameter-training-free, not label-free.
- Primary AVeriTeC results use Jaccard-0.90 grouped fold exclusion: 343 routes, not the historical exact-fold 357-route result.
- Fold exclusion protects grouped claim identity, not shared evidence/source overlap.
- SciFact projected and raw four-label results are separate analyses.

## Headline checks

| Condition | Check |
|---|---|
| AVeriTeC grouped R6 | 90 helpful - 44 harmful = 46; 46/3068 = 0.0149935 accuracy gain |
| CLIMATE-FEVER R6 | 53 - 28 = 25; 25/1535 = 0.0162866 |
| SciFact oracle projected R6 | 11 - 2 = 9; 9/300 = 0.0300 |
| SciFact BM25 R6 | 9 - 4 = 5; 5/300 = 0.016667 |
| Forward-pass cost | `4 + 4 * route_rate` in all four-label conditions |

## Required server archive

The release archive should include normalized inputs, B1/A3 predictions, memories, routes, matched-budget summaries, significance JSON, and manifests with SHA-256 checksums. Exact known paths and claim mappings are recorded in `paper/claims_to_artifacts.csv`.
