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
