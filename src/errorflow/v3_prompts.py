"""Frozen V3 prompt portfolio.

The B1 prompt is duplicated intentionally so its byte-level contract can be
checked without changing the historical V2 runner implementation.
"""

import hashlib


B1_CONFLICT_AUDIT = (
    "Audit whether the evidence contains credible support and credible contradiction for the same essential claim. "
    "Choose CONFLICTING only when both directions are genuinely present or the claim cherry-picks a mixed record. "
    "If evidence supports one direction without a credible opposite, choose SUPPORTED or REFUTED. "
    "If neither direction is established, choose NOT ENOUGH INFO."
)
B1_CONFLICT_AUDIT_SHA256 = "700ebdc1b55cb9f04f7524f31b7a5d3bd41813f198e5629fe75ca551333decf7"


V3_ACTION_PROMPTS = {
    "coverage_audit_v2": (
        "Check every essential entity, relation, condition, and subclaim separately. "
        "SUPPORTED requires complete grounding of all essential parts. REFUTED requires direct evidence that an essential part is false. "
        "Use NOT ENOUGH INFO for missing links, partial support, or unverified subclaims. "
        "Use CONFLICTING only when credible evidence genuinely supports and contradicts the same essential claim."
    ),
    "refutation_audit_v2": (
        "Test whether the evidence explicitly contradicts an essential claim component rather than merely failing to support it. "
        "Check negation, counterexamples, and incompatible facts. Choose REFUTED only for established contradiction; "
        "choose NOT ENOUGH INFO for absence or insufficiency. Choose CONFLICTING only when both support and contradiction are established."
    ),
    "conflicting_rescue_v1": (
        "Independently list the strongest supporting evidence and the strongest refuting evidence for the same essential claim. "
        "Choose CONFLICTING only if both directions are credible and materially established, including genuine cherry-picking. "
        "Multiple sources alone are not a conflict. If only one direction is established choose SUPPORTED or REFUTED; otherwise choose NOT ENOUGH INFO."
    ),
    "entity_scope_audit_v1": (
        "Verify exact entity identity, time period, geographic scope, numerical quantities, and universal or existential quantifiers. "
        "A verdict must follow the scope of the claim, not a nearby entity or broader statement. "
        "Choose SUPPORTED or REFUTED only when the aligned evidence establishes that verdict; otherwise choose NOT ENOUGH INFO, "
        "or CONFLICTING when aligned credible evidence establishes both directions."
    ),
    "structured_decomposition_v1": (
        "Decompose the claim into atomic propositions. For each proposition, check whether aligned evidence supports it, refutes it, "
        "or leaves it unresolved. Aggregate conservatively: all essential propositions must be supported for SUPPORTED; "
        "an established false essential proposition permits REFUTED; unresolved essentials require NOT ENOUGH INFO; "
        "use CONFLICTING only for material credible support and contradiction concerning the same essential proposition."
    ),
}


# Ontology-only SciFact sensitivity prompts. These are intentionally separate
# from the immutable four-label V3 prompts: SciFact has no CONFLICTING verdict,
# so mixed support/refutation is resolved to NOT ENOUGH INFO. Results produced
# with these prompts are not interchangeable with post-hoc label projection.
SCIFACT_THREE_LABEL_PROMPTS = {
    "conflict_audit_v1": (
        "Audit whether the evidence credibly supports or contradicts the same essential claim. "
        "Choose SUPPORTED when support is established without a credible contradiction, and REFUTED "
        "when contradiction is established without credible support. Choose NOT ENOUGH INFO when "
        "neither direction is established or when the available evidence is materially mixed."
    ),
    "conflicting_rescue_v1": (
        "Independently identify the strongest supporting and refuting evidence for the same essential claim. "
        "Choose SUPPORTED if only material support is established, REFUTED if only material contradiction "
        "is established, and NOT ENOUGH INFO if neither direction or both directions are established."
    ),
}


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def assert_b1_immutable() -> None:
    actual = prompt_sha256(B1_CONFLICT_AUDIT)
    if actual != B1_CONFLICT_AUDIT_SHA256:
        raise RuntimeError(f"B1 prompt changed: expected {B1_CONFLICT_AUDIT_SHA256}, got {actual}")
