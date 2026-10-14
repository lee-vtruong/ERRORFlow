# Paper revision round 3: similarity-component ablation

Date: 2026-10-14. Rendering intentionally deferred.

## Integrated evidence

- Added the matched-budget A0/A2/A3/A5/A6/A7/A8 table to the main paper.
- Kept A2 and A7 despite their higher macro-F1 because inclusion follows
  mechanistic relevance, not whether a result favors Full R6.
- Moved A1, A4, natural-threshold comparisons, and fold-direction summaries
  to the appendix.
- Reported exact McNemar tests with Holm correction over one family of eight
  comparisons per mode and paired bootstrap intervals.
- Kept A8 seed 2026 distinct from the descriptive 20-seed mean.
- Replaced the obsolete statement that component ablations were unexecuted.
- Added the explicit discussion limitation that individual necessity and
  superiority of manually assigned weights are not established.

## Claim boundary

The ablation supports the value of similarity ranking over random compatible
neighbors and rejects evidence-length-only routing as sufficient. It does not
show that claim similarity is independently necessary, that label-score
similarity always improves macro-F1, or that 0.35/0.50/0.15 is better than
uniform weighting.
