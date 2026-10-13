# ERRORFlow V3 abstract and contribution draft

Status: **PAPER DRAFT — CLAIMS BOUNDED BY COMPLETED V3 EVIDENCE**.

## Abstract

Large language model fact verifiers can benefit from a second inference pass,
but applying correction prompts uniformly increases compute and may regress
already-correct predictions. We introduce ERRORFlow, a training-free residual
correction framework that stores source-domain action outcomes in a
non-parametric error memory and selectively invokes a frozen corrective prompt
when similar residual patterns indicate positive utility. ERRORFlow updates no
model weights and explicitly measures the quality-cost trade-off. Using
Qwen3-4B-Instruct-2507, conservative grouped evaluation on AVeriTeC shows that
memory-conditioned routing improves accuracy by 1.50 points and macro-F1 by
1.17 points while routing 11.18% of examples. On the locked four-label
CLIMATE-FEVER evaluation, always-on correction improves accuracy and macro-F1
by 3.06 and 5.85 points, while selective ERRORFlow obtains smaller significant
gains at 19.93% routing. On SciFact with open-corpus BM25 evidence, ERRORFlow
routes 18.67%, improving macro-F1 by 2.58 points with a positive paired
bootstrap interval, although its accuracy interval crosses zero. Retrieval
stratification shows that selection avoids the regressions caused by always-on
correction when relevant evidence is missed. Matched-budget comparisons do not
establish superiority over simple confidence routing across external domains,
and conflicting-evidence classification remains weak. These findings support
selective residual correction as a compute-efficient inference strategy while
highlighting the limits of source-memory transfer and ontology-dependent
evaluation.

## Research questions

- RQ1: Can a fixed second-pass action improve a frozen fact verifier without
  parameter updates?
- RQ2: Can source residual memory identify examples where that action has
  positive expected utility?
- RQ3: Does selective routing retain gains at lower inference cost than
  always-on correction?
- RQ4: Do gains transfer across label ontologies and evidence conditions?
- RQ5: Does memory-conditioned routing outperform simpler matched-budget
  uncertainty and similarity controls?

## Contributions

1. A reproducible training-free residual correction pipeline with immutable
   B1/A3 prompts, versioned action memory, and deterministic routing.
2. Leakage-audited, fold-excluded source evaluation with near-duplicate-group
   sensitivity and exact artifact replay.
3. Matched-budget routing comparisons separating action quality from routing
   quality and compute opportunity.
4. Locked cross-domain evaluations on CLIMATE-FEVER and SciFact, including
   explicit label-projection sensitivity.
5. An open-corpus retrieved-evidence study that separates retrieval hits,
   retrieval misses, and NEI behavior.

## Claims to avoid in the abstract

- Do not claim state of the art.
- Do not call the method label-free or supervision-free.
- Do not claim that error memory significantly beats confidence routing on
  every external dataset.
- Do not claim robust CONFLICTING-label detection.
- Do not present projected SciFact oracle scores without the raw sensitivity
  and projection counts.
- Do not claim an independently validated cost-aware ranking advantage under a
  single constant-cost action.
