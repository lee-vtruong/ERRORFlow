# ERRORFlow V3 Reviewer Round 4 Results

## Group-dependent AVeriTeC uncertainty

Near-duplicate grouping at token-Jaccard 0.90 produced 2,924 groups from
3,068 claims (largest group: 9). Memory-Global improved Base from 0.74022 to
0.75847 accuracy and from 0.54602 to 0.56657 macro-F1.

- Accuracy delta: +0.01825; group-bootstrap 95% CI [0.01102, 0.02560].
- Macro-F1 delta: +0.02055; group-bootstrap 95% CI [0.00901, 0.03255].
- Exact McNemar: 38 Base-only correct versus 94 Memory-only correct,
  two-sided p = 1.186e-6.

At the same 375-route budget, outcome-balanced logistic tied Memory-Global
accuracy at 0.75847 and reached 0.55342 macro-F1 versus 0.56657. For
Logistic-minus-Memory, the group-bootstrap accuracy CI was
[-0.00653, 0.00650] and the macro-F1 CI was [-0.02678, 0.00006]. The result
does not establish a difference at 95%; it supports treating memory and
supervised gating as competitive implementations of outcome-based gating.

## SciFact conventional three-label Base control

The post-hoc conventional prompt control was evaluated on the same 300 dev
claims as the native three-label audit-style Base.

| Evidence | Audit Base Acc. | Standard Base Acc. | Delta Acc. (95% CI) | Audit F1 | Standard F1 | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Oracle rationales | 0.4900 | 0.8600 | +0.3700 [0.3100, 0.4300] | 0.3767 | 0.8447 | 2.306e-25 |
| BM25 top-5 | 0.5800 | 0.6533 | +0.0733 [0.0233, 0.1233] | 0.5387 | 0.6299 | 0.00622 |

These controls invalidate a broad interpretation of the earlier SciFact
routing results as evidence that the dataset intrinsically favors abstention.
They instead show that the audit-style prompt pair was a major confound. The
existing SciFact routing and harm-containment outputs are retained only as
prompt-sensitivity diagnostics. A fair SciFact routing test would need a new
frozen alternative action and source outcome memory built around the standard
Base. Because SciFact dev had already been inspected, this is not presented as
a new untouched held-out result.
