# ERRORFlow V2 — AVeriTeC one-shot holdout report

## Protocol

- Dataset: official AVeriTeC dev examples 25–499, `n=475`.
- Evidence setting: oracle QA evidence; this is verifier/routing evaluation,
  not end-to-end web retrieval.
- Frozen verifier: Qwen3-4B-Instruct-2507, S0 mean label-log-probability scoring.
- Memory: 465 action outcomes from 155 train-only examples.
- Router: `averitec_errorflow_router_v1`, frozen before holdout candidates.
- Comparators fixed before metric inspection: S0 baseline, ERRORFlow, and
  standalone conflict audit.
- Statistics: paired bootstrap, 5,000 samples, seed 2026.

## Main results

| System | Accuracy | Macro-F1 | Latency (ms) | Forward passes |
|---|---:|---:|---:|---:|
| S0 baseline | 0.68632 | 0.52417 | 114.69 | 4.000 |
| ERRORFlow V1 | 0.72421 | 0.55839 | 158.85 | 5.406 |
| Conflict audit standalone | **0.75789** | **0.56392** | **120.12** | **4.000** |

ERRORFlow versus baseline:

- Accuracy delta: +0.03789, CI95% `[+0.00842, +0.06947]`.
- Macro-F1 delta: +0.03422, CI95% `[-0.00288, +0.07245]`.
- Helpful/harmful/net: 38/20/+18.
- Route rate: 35.16% (167/475).

Conflict audit versus baseline:

- Accuracy delta: +0.07158, CI95% `[+0.03789, +0.10737]`.
- Macro-F1 delta: +0.03975, CI95% `[-0.00941, +0.08730]`.
- Helpful/harmful/net: 56/22/+34.

ERRORFlow versus conflict audit:

- Accuracy delta: -0.03368, CI95% `[-0.06316, -0.00632]`.
- Macro-F1 delta: -0.00553, CI95% `[-0.04953, +0.04249]`.

## Interpretation

The core training-free claim is supported in a limited form: train-only error
memory and selective routing improved frozen-baseline accuracy on an untouched
holdout. The gain was not a random-routing artifact and generalized from the
train screen, although at a smaller magnitude (+3.79 rather than +6.30 points).

The stronger conclusion is negative but useful: memory routing did not beat a
well-chosen global instruction. Standalone conflict audit had significantly
higher accuracy, nearly identical macro-F1, and lower practical inference cost.
Therefore V1 should not be presented as the best deployed system.

ERRORFlow did retain better rare-class balance: CONFLICTING F1 was 0.23256
versus 0.10 for conflict audit. This did not yield a macro-F1 advantage on 475
examples, and CONFLICTING recall remains poor in every system.

## Cost interpretation

As a baseline-plus-intervention cascade, always-conflict costs eight forward
passes and ERRORFlow saves work by routing only 35.16%. As a standalone prompt
replacement, however, conflict audit costs four passes and dominates ERRORFlow
on both accuracy and runtime. Both views must be reported; only the latter is a
strict practical system comparison.

## Research decision

Close V1 without holdout tuning. A revised method may promote the conflict
instruction to the new frozen baseline and learn only residual corrections,
but it must be developed with train-only nested validation. This 475-example
holdout cannot be reused as an unbiased selection set. Confirmation should use
an untouched dataset/split, for example cross-dataset evaluation after the new
policy is frozen.
