# V3 in-domain memory and multi-action extension protocol

This extension is isolated from the frozen ERRORFlow V3 method and outputs.
Its purpose is to distinguish source-memory mismatch from a general failure of
selective escalation, then test heterogeneous actions and costs. Existing
AVeriTeC, CLIMATE-FEVER, and SciFact results must not be overwritten.

## Phase I: in-domain outcome memory

For each dataset, cache Base and Alt for every source example, construct paired
helpful/harmful/neutral outcome records, and evaluate with five group-safe
outer folds. A query's complete fold and near-duplicate group must be excluded
from usable memory. Report Base, Alt-All, confidence, random, logistic,
Memory-SameLabel, and Memory-Global at the same route counts.

- CLIMATE-FEVER: treat the 1,535 released claims as a grouped development
  population. This is internal cross-validation, not external confirmation.
- SciFact: use official train claims to construct memory and dev only for one
  frozen evaluation. Run oracle-rationale and open-corpus BM25 evidence as
  separate protocols; never build memory from dev labels.
- Keep label adapters outside the router. For SciFact, project CONFLICTING to
  NEI only after routing, exactly as in the frozen evaluation.

Required outputs per condition:

1. fold assignments and duplicate groups;
2. Base/Alt prediction caches and immutable manifests;
3. residual memory with source split and fold provenance;
4. prediction-level rows for every control;
5. route count, helpful/harmful/net, per-class F1, average passes;
6. exact McNemar, paired bootstrap intervals, and random-seed variability;
7. quality--budget curves and the fraction of oracle helpful gain captured.

The decisive comparison is in-domain Memory-Global versus Alt-All. If routing
retains most of Alt-All's gain with substantially fewer passes, the transfer
failure can be attributed partly to source-memory mismatch. If not, the paper
must report that selective escalation is useful only when the alternative
action has mixed utility.

## Phase II: heterogeneous corrective actions

Do not begin model selection until Phase I is frozen. Define two or three
immutable alternative prompts before inspecting their routed results, for
example conflict audit, evidence-coverage audit, and refutation audit. Each
action must have:

- a fixed prompt hash and label-scoring protocol;
- cached predictions on every source example;
- a declared forward-pass or latency cost;
- action-specific helpful/harmful/neutral outcomes;
- action-compatible memory retrieval without target labels.

At inference, estimate one utility per action and choose among Base and the
eligible actions. Compare against Alt-All for each action, the best single
always-on action, random action choice, confidence routing, and an oracle
action selector. Report both a matched total-forward-pass budget and natural
cost-sensitive decisions. With heterogeneous costs, the cost term can affect
action ranking rather than merely shift a common threshold.

This phase changes the research question and is not a patch to the current
single-action result. It should be presented as an extension only after the
single-action evidence and in-domain memory results are complete.
