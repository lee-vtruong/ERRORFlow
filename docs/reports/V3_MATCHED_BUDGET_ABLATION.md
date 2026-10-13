# ERRORFlow V3 matched-budget routing ablation

Status: **IMPLEMENTED; SERVER CACHED-ACTION EXECUTION PENDING**.

## Confirmatory question

Does structured residual memory add correction value beyond uncertainty,
label-conditional, similarity-only, static-pattern, and random routing when all
methods receive the same opportunity to invoke cached A3?

## Systems

- `B1`: no correction;
- `A3-All`: correction on every example;
- `R-Random`: 20 deterministic seeds;
- `R-Conf`: lowest maximum B1 label probability;
- `R-Margin`: smallest B1 top-two probability margin;
- `R-Label`: margin routing separately within each predicted label;
- `R-Sim`: highest similarity to compatible source memory, ignoring outcomes;
- `R-Static`: utility of a coarse B1-label × evidence-count rule;
- `R-Memory`: local recovery-minus-regression without the cost penalty;
- `R6`: local recovery-minus-regression-minus-cost, restricted to positive
  estimated utility.

The evaluator uses the conservative token-Jaccard 0.90 connected-component
folds established after WP0. Every memory-derived score excludes the complete
query fold. The A3 prediction cache is shared by every router.

## Budgets and measurements

Requested route budgets are `0/5/10/11.64/15/20/35/50/100%`, plus the exact
natural positive-utility R6 rate induced by the conservative folds. R6 may
route fewer examples than a requested budget because it never admits
non-positive utility. Random routing reports 20 seeds and empirical seed
variation.

Metrics include accuracy, macro-F1, per-class F1, helpful/harmful/net flips,
forward passes, generated tokens, and cached-component mean/p95 latency.
Tokenizer-level input tokens were not recorded historically; the evaluator
reports reprocessed claim/evidence whitespace tokens separately and leaves
model tokenizer input tokens null rather than fabricating them.

Primary comparison is R6 against the strongest deterministic non-memory router
at the natural R6 route count. Exact McNemar, paired bootstrap, and Holm-adjusted
McNemar p-values are produced. Because A3 was selected on an AVeriTeC train
screen, this remains `VERIFIED_BUT_TRAIN_SELECTED` evidence even with strict
fold isolation.
