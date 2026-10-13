# Internal review report

## Recommendation

Promising workshop/conference submission after artifact packaging and venue-specific formatting. The paper supports a narrow claim: source-outcome memory can route a frozen corrective instruction efficiently, with positive source-internal and external point estimates. It does not establish a universally superior router or projection-independent cross-domain transfer.

## Strengths

- Exact implementation-based method specification and explicit forward-pass accounting.
- Helpful/harmful accounting exposes regressions hidden by aggregate metrics.
- Conservative near-duplicate grouping, exact replay, and target-gold invariance checks.
- Matched route budgets and negative results are reported.
- Open-corpus SciFact condition separates retrieval misses from inference errors.

## Major concerns addressed in this revision

1. The earlier draft had no citations or bibliography. It now cites 21 primary sources with an automated consistency check.
2. Historical and conservative AVeriTeC results were easy to conflate. The paper now uses only the 343-route grouped result as primary.
3. SciFact gains could be overstated. Raw and projected results and projection counts are now explicit.
4. “Training-free” could imply no supervision. The paper now consistently says parameter-training-free and documents source-label use.
5. The repository lacked a claim ledger. It now maps claims to server artifacts and tracks a figure-generation snapshot.

## Remaining concerns

- The raw prediction archive is not in the checkout, so an external reviewer cannot recompute all intervals yet.
- One model and one corrective action limit generality.
- A3 and router selection are not nested with respect to AVeriTeC train.
- CONFLICTING remains poorly recognized.
- External R6 advantages over matched simple routers are inconclusive.
- The model identifier is hashed, but weight/tokenizer revisions are not cryptographically pinned.

## Claim-language check

Allowed: “improves B1 under the reported protocol,” “selective harm avoidance,” and “positive macro-F1 interval.” Avoid: “state of the art,” “label-free,” “causal diagnosis,” “calibrated confidence,” “universal transfer,” and “superior to confidence routing.”
