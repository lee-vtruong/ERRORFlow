# ERRORFlow V3 — Standard Base Decisive Experiment Results

## Outcome

The predeclared success criterion was not met on either AVeriTeC or
CLIMATE-FEVER. The frozen Standard Base/Standard Alt pair was evaluated with
newly rebuilt outcome memories; no historical audit-pair memory was accepted.

| Dataset | Standard Base Acc. | Standard Alt-All Acc. | Memory-Global Acc. | Memory-Global F1 | Routes | Helpful/Harmful | Delta Acc. (group CI95) |
|---|---:|---:|---:|---:|---:|---:|---:|
| AVeriTeC | 76.86 | 65.68 | 76.69 | 56.46 | 28/3,068 | 2/7 | -0.16 pp [-0.36, +0.03] |
| CLIMATE-FEVER | 57.39 | 56.81 | 58.05 | 45.04 | 153/1,535 | 36/26 | +0.65 pp [-0.33, +1.69] |

Exact McNemar p-values for Memory-Global versus Standard Base are .180 on
AVeriTeC and .253 on CLIMATE-FEVER.

## Interpretation

- **AVeriTeC:** Standard Alt is strongly non-complementary (39 helpful versus
  382 harmful changes). Memory-Global routes 0.91% and still produces net -5,
  reducing both accuracy and macro-F1 relative to Standard Base.
- **CLIMATE-FEVER:** Memory-Global filters the action to net +10 and gives a
  positive point estimate, but both grouped confidence intervals include zero.
  At the same 153-route budget, Memory-SameLabel reaches 58.31% accuracy and
  45.22% macro-F1, above Memory-Global's point estimates.
- **Scientific scope:** outcome-guided routing is not rejected as a general
  research direction, but its utility depends on a sufficiently complementary
  Base/Alt action pair. The audit-pair gains cannot be generalized to a
  stronger conventional Base.

The CLIMATE-FEVER result here is a five-fold in-domain grouped cross-fit over
the only labeled release. It is not the frozen AVeriTeC-memory transfer result
reported in the main experiments.

## Reporting decision

- Keep the audit-pair experiments in the main results with explicit scope.
- Move the complete Standard Base/Alt control to the appendix.
- State in limitations and conclusion that improvement over stronger prompt
  pairs is not established.
- Preserve all manifests, predictions, fold-safe memory records, and
  statistical outputs in the artifact.
- Do not tune another prompt or router on these inspected populations and
  present it as confirmatory.
