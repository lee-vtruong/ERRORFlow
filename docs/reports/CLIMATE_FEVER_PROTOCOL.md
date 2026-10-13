# CLIMATE-FEVER external validation protocol

Status: **PROTOCOL AND DATA MANIFEST VERIFIED; INFERENCE PENDING**.

## Dataset reconnaissance

The official CLIMATE-FEVER v1.0.1 release contains 1,535 real-world climate
claims and five Wikipedia evidence candidates per claim (7,675 pairs). It is a
single labeled release, not an official train/development/test partition. The
claim-label distribution reported by the paper is 655 SUPPORTS, 253 REFUTES,
474 NOT_ENOUGH_INFO, and 153 DISPUTED.

The downloaded official v1.0.1 release contains 654 SUPPORTS, 253 REFUTES, 474
NOT_ENOUGH_INFO, and 154 DISPUTED claims. This one-example difference from the
paper is retained as a release-version discrepancy rather than manually
altered. The release has 1,535 unique claim IDs, exactly five evidence
candidates for every claim, 7,675 evidence pairs, and SHA-256
`8a4b9032d861be482ffb49dddfd283ffa6089e654f1e968040011882c5eb6e0b`.

The five evidence sentences were retrieved by the authors' evidence candidate
retrieval system and then annotated. They are therefore neither independently
retrieved by ERRORFlow nor unrestricted oracle rationales. This experiment is
reported as **verification over official retrieved-and-annotated candidates**.
Evidence labels, votes, and entropy are excluded from the normalized prompt
artifact; only sentence text and article title provenance are retained.

The official repository and dataset card do not declare a resolvable license.
ERRORFlow will not redistribute the raw artifact and will label the license as
unknown rather than infer permission from third-party mirrors.

Sources:

- https://github.com/tdiggelm/climate-fever-dataset
- https://github.com/tdiggelm/climate-fever-dataset/releases/tag/1.0.1
- https://arxiv.org/abs/2012.00614

## Label ontology

The official aggregation rule defaults to NOT_ENOUGH_INFO, selects SUPPORTS or
REFUTES when evidence establishes one direction, and selects DISPUTED when both
supporting and refuting evidence are present. This definition directly matches
the implemented B1 meaning of CONFLICTING more closely than the SciFact case.
The adapter is frozen before evaluation as:

```text
SUPPORTS        -> SUPPORTED
REFUTES         -> REFUTED
NOT_ENOUGH_INFO -> NOT ENOUGH INFO
DISPUTED        -> CONFLICTING
```

No post-hoc projection is planned. Per-class F1 for DISPUTED/CONFLICTING is a
required result because it tests the conflict-first premise directly.

## Leakage and split policy

All 1,535 claims form one locked external evaluation population. There is no
target training or threshold selection. Gold claim labels and evidence-level
annotations are unavailable to B1, A3, memory retrieval, and routing; they are
read only during final metric computation. The complete AVeriTeC source memory
is permitted because no CLIMATE-FEVER outcome enters it.

The absence of an official held-out split means this experiment must remain a
one-shot external evaluation. Any later target adaptation would require a new
partition and would forfeit the untouched-external interpretation.

## Frozen systems

The evaluation will compare B1, A3-All, matched-budget confidence-only,
matched-budget similarity-only, and frozen source-memory R6. All selective
routers reuse identical cached B1 and A3 outputs. Confidence and similarity
controls will be matched to the natural R6 route count without target labels.

Primary metrics are accuracy and four-class macro-F1. Secondary metrics are
per-class F1, route rate, helpful/harmful/net corrections, exact McNemar,
paired bootstrap intervals, and model forward passes.

## Gate

The structural gate passed: 1,535 unique claims, five candidates per claim, no
evidence annotation metadata in the normalized prompt artifact, and a pinned
official-release hash. The paper/release one-label discrepancy is documented
above and does not block inference.
