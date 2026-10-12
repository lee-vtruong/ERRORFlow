# ERRORFlow V3 counterfactual actions

Status: **750/750 GPU SCREEN PREDICTIONS COMPLETED; UTILITY SUMMARY PENDING**.

B1 remains immutable. Five correction actions are available through both the
historical runner and the hash-validated V3 runner:

- `coverage_audit_v2`
- `refutation_audit_v2`
- `conflicting_rescue_v1`
- `entity_scope_audit_v1`
- `structured_decomposition_v1`

All use the frozen Qwen model, identical four-label mean log-probability
scoring, four forward passes, and no generated tokens. The V3 runner records
prompt, model-identifier, evidence, generation-config, and composite cache
hashes. It supports dry run, sample/call/runtime bounds, strict resume, and
config-manifest validation.

The first experiment is a transition-stratified train-only screen including
correct controls. Full 3,068 × 5 inference is prohibited until heterogeneous
action utility is observed on this bounded screen.

## Completed bounded screen

The server completed five actions on the same 150-example screen, producing
150 predictions per action and 750 total action outcomes. The screen spans all
observed B1 error transitions and correct-control labels across their margin
ranges.

The next step computes helpful, harmful, changed, and net outcomes by action
and transition. Group weights map the screen back to the 3,068-example B1 train
population for descriptive screening only. Because the screen is deterministic
and margin-spanning rather than randomly sampled, weighted results are not
holdout estimates and cannot support a generalization claim.

## Screen utility result

Only `conflicting_rescue_v1` has positive population-weighted net utility:
117.8 projected examples, projected accuracy delta +0.03840, and projected
macro-F1 delta +0.01731. The other four actions have negative weighted net
utility despite positive unweighted screen net values. Their regressions occur
in large correct-control populations, so raw screen counts are misleading.

`conflicting_rescue_v1` recovered 23 and regressed 7 screen examples. Its main
observed gains came from REFUTED→NEI and REFUTED→SUPPORTED residuals; it did not
recover the sampled CONFLICTING→REFUTED or CONFLICTING→SUPPORTED errors. Action
names are therefore not treated as validated causal mechanisms.

This result advances `conflicting_rescue_v1` as the always-action and
confidence-routing comparator. All five actions remain available to the
fold-excluded memory/utility router because some globally harmful actions have
positive utility in narrower observable neighborhoods. Gold transitions are
offline analysis fields and are forbidden as online routing inputs.

## Full-train A3 cache

The bounded-screen result authorized one full action run before expanding the
entire portfolio. `conflicting_rescue_v1` completed all 3,068 AVeriTeC train
examples in 366.998 seconds with 12,272 model forward passes. The run manifest
records model hash `ff40807e...fb0da`, input SHA-256
`f5d4c48e...570f3b`, prompt SHA-256 `09e187ec...97aaf`, and generation-config
hash `89c39128...cf7`.

Full-train helpful/harmful measurement and fixed-policy fold-excluded routing
remain pending. The other four full action runs are not yet authorized.
