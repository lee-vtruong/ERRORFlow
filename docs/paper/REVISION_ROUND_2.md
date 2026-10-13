# Paper revision round 2: experimental completeness and claim calibration

Date: 2026-10-14. Rendering intentionally deferred at the author's request.

## Completed source changes

1. Reorganized the manuscript around Problem Formulation, Methodology,
   Experimental Setup, Main Results, Analysis and Ablations, Discussion,
   Limitations, and Conclusion.
2. Added paired 95% bootstrap intervals for R6 versus B1 in all four reported
   evaluation conditions. The abstract, result text, table, and cross-domain
   figure caption now distinguish positive intervals from intervals crossing
   zero.
3. Replaced the cross-domain delta bar-chart generator with a forest-plot
   generator that displays uncertainty and marks SciFact oracle as
   adapter-sensitive.
4. Added a full CLIMATE-FEVER per-class F1 table for B1, A3-All, and R6. The
   available AVeriTeC per-class values are labeled as historical exact-fold
   sensitivity results; conservative grouped-primary values are not inferred.
5. Added the frozen development-fold R6 budget sweep and documented saturation
   at the natural positive-utility route set.
6. Added memory-size and near-duplicate grouping sensitivity. Missing
   claim-only, score-only, length-only, no-outcome, no-regression, and parameter
   grid experiments are explicitly listed as unexecuted rather than fabricated.
7. Added compute accounting for B1, memory lookup, routing, A3, input volume,
   and the complete pipeline. The manuscript claims forward-pass efficiency
   only; it does not claim a measured wall-clock or monetary reduction.
8. Added a protocol-aware published-system table that separates internal
   AVeriTeC, CLIMATE-FEVER candidates, SciFact oracle, and SciFact open-corpus
   BM25. No incompatible scores are ranked together.
9. Calibrated terminology throughout: ERRORFlow performs memory-guided
   selection of one fixed corrective action and does not learn new correction
   strategies or update model parameters.

## Provenance boundaries

- AVeriTeC grouped-primary results and confidence intervals come from the
  grouped 0.90 matched-budget summary.
- AVeriTeC per-class and budget-sweep values come from the earlier exact-fold
  protocol and are marked with that protocol in the paper.
- CLIMATE-FEVER and SciFact numbers come from their frozen external summaries.
- The 366.998-second A3 cache runtime is a component batch measurement, not an
  end-to-end ERRORFlow benchmark.

## Deferred work

- Regenerate all derived figures, including the revised uncertainty plot.
- Compile the LNCS manuscript and inspect every rendered page.
- Insert the genuine worked trace after exporting it from the server.
- Replace anonymous author, affiliation, funding, and acknowledgment fields.
