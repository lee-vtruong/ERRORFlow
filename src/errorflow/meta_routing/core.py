"""Leakage-safe feature extraction and deterministic meta-routing policies.

This module never reads gold labels or candidate predictions.  Those values are
used only by the evaluation layer after all routes have been materialized.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass

from errorflow.error_memory import TOKEN_RE, score_distribution
from errorflow.v3_memory import ResidualMemoryStore
from errorflow.v3_routing import estimate_action


CANONICAL_TOKENIZER = TOKEN_RE.pattern


@dataclass(frozen=True)
class NeighborhoodSummary:
    compatibility: str
    eligible: bool
    ineligibility_reason: str | None
    support: int
    utility: float
    recovery: float
    regression: float
    expected_cost: float
    signed_mean: float
    signed_variance: float
    effective_sample_size: float
    standard_error: float
    mean_similarity: float
    max_similarity: float
    similarity_margin: float
    same_label_count: int
    cross_label_count: int
    cross_label_similarity_mass: float
    neighbor_ids: tuple[str, ...]
    neighbor_claim_ids: tuple[str, ...]
    neighbor_similarities: tuple[float, ...]


@dataclass(frozen=True)
class MetaFeatures:
    example_id: str
    baseline_prediction: str
    baseline_entropy: float
    baseline_top_two_margin: float
    hard: NeighborhoodSummary
    nlf: NeighborhoodSummary
    utility_difference_nlf_minus_hard: float
    top5_neighbor_overlap: float
    hard_lookup_ms: float
    nlf_lookup_ms: float

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PolicyDecision:
    policy: str
    route: bool
    score: float
    selected_neighborhood: str | None
    reason: str

    def to_dict(self):
        return asdict(self)


def _finite(value: float, fallback: float = -math.inf) -> float:
    return value if math.isfinite(value) else fallback


def _utility_difference(left: float, right: float) -> float:
    if math.isfinite(left) and math.isfinite(right):
        return left - right
    if math.isfinite(left):
        return math.inf
    if math.isfinite(right):
        return -math.inf
    return 0.0


def _weighted_summary(neighbors, action, query_label, min_support, cost_weight, compatibility):
    rows = [(record, float(weight)) for record, weight in neighbors
            if record.correction_strategy == action]
    if not rows:
        return NeighborhoodSummary(
            compatibility, False, "no_eligible_neighbors", 0, -math.inf,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.inf,
            0.0, 0.0, 0.0, 0, 0, 0.0, (), (), (),
        )
    weights = [max(0.0, weight) for _, weight in rows]
    if sum(weights) <= 0.0:
        weights = [1.0] * len(rows)
    total = sum(weights)
    squared = sum(weight * weight for weight in weights)
    signed = [record.recovery_count - record.regression_count for record, _ in rows]
    signed_mean = sum(weight * value for weight, value in zip(weights, signed)) / total
    variance = sum(
        weight * (value - signed_mean) ** 2
        for weight, value in zip(weights, signed)
    ) / total
    n_eff = total * total / squared if squared > 0 else 0.0
    standard_error = math.sqrt(max(0.0, variance) / max(n_eff, 1e-12))
    weighted_neighbors = [(record, weight) for (record, _), weight in zip(rows, weights)]
    estimate = estimate_action(action, weighted_neighbors, cost_weight=cost_weight)
    similarities = [float(score) for _, score in rows]
    ordered = sorted(similarities, reverse=True)
    same = sum(
        record.retrieval_features.get("baseline_prediction") == query_label
        for record, _ in rows
    )
    cross_mass = sum(
        score for record, score in rows
        if record.retrieval_features.get("baseline_prediction") != query_label
    )
    eligible = estimate.support >= min_support
    return NeighborhoodSummary(
        compatibility=compatibility,
        eligible=eligible,
        ineligibility_reason=None if eligible else "insufficient_support",
        support=estimate.support,
        utility=estimate.utility if eligible else -math.inf,
        recovery=estimate.expected_recovery,
        regression=estimate.expected_regression,
        expected_cost=estimate.expected_cost,
        signed_mean=signed_mean,
        signed_variance=variance,
        effective_sample_size=n_eff,
        standard_error=standard_error,
        mean_similarity=sum(similarities) / len(similarities),
        max_similarity=max(similarities),
        similarity_margin=(ordered[0] - ordered[1]) if len(ordered) > 1 else ordered[0],
        same_label_count=same,
        cross_label_count=len(rows) - same,
        cross_label_similarity_mass=cross_mass,
        neighbor_ids=tuple(record.memory_id for record, _ in rows),
        neighbor_claim_ids=tuple(record.source_claim_id for record, _ in rows),
        neighbor_similarities=tuple(similarities),
    )


def _entropy_and_margin(label_scores):
    probabilities = score_distribution({key: float(value) for key, value in label_scores.items()})
    if not probabilities:
        return 0.0, 0.0
    values = sorted(probabilities.values(), reverse=True)
    entropy = -sum(value * math.log(max(value, 1e-300)) for value in values)
    margin = values[0] - values[1] if len(values) > 1 else values[0]
    return entropy, margin


def build_meta_features(
    example_id,
    baseline_row,
    store: ResidualMemoryStore,
    *,
    action="conflicting_rescue_v1",
    top_k=5,
    min_support=2,
    cost_weight=0.1,
    exclude_claim_ids=None,
    exclude_folds=None,
):
    """Build pre-action features for one query under HardLabel and NLF."""
    label_scores = baseline_row.get("label_scores") or baseline_row.get("metadata", {}).get("label_scores", {})
    evidence_tokens = baseline_row.get(
        "evidence_tokens", baseline_row.get("metadata", {}).get("evidence_tokens", 0)
    )
    query = {
        "example_id": str(example_id),
        "claim": baseline_row["claim"],
        "retrieval_features": {
            "baseline_prediction": baseline_row["prediction"],
            "label_scores": label_scores,
            "evidence_tokens": int(evidence_tokens or 0),
        },
    }
    common = {
        "top_k": top_k,
        "action": action,
        "exclude_claim_ids": set(exclude_claim_ids or ()),
        "exclude_folds": set(exclude_folds or ()),
    }
    started = time.perf_counter()
    hard_neighbors = store.retrieve(query, compatibility_mode="same_label", **common)
    hard_lookup_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    nlf_neighbors = store.retrieve(query, compatibility_mode="none", **common)
    nlf_lookup_ms = (time.perf_counter() - started) * 1000.0
    hard = _weighted_summary(
        hard_neighbors, action, baseline_row["prediction"], min_support, cost_weight, "same_label"
    )
    nlf = _weighted_summary(
        nlf_neighbors, action, baseline_row["prediction"], min_support, cost_weight, "none"
    )
    hard_ids, nlf_ids = set(hard.neighbor_ids), set(nlf.neighbor_ids)
    union = hard_ids | nlf_ids
    overlap = len(hard_ids & nlf_ids) / len(union) if union else 1.0
    entropy, margin = _entropy_and_margin(label_scores)
    return MetaFeatures(
        example_id=str(example_id),
        baseline_prediction=baseline_row["prediction"],
        baseline_entropy=entropy,
        baseline_top_two_margin=margin,
        hard=hard,
        nlf=nlf,
        utility_difference_nlf_minus_hard=_utility_difference(nlf.utility, hard.utility),
        top5_neighbor_overlap=overlap,
        hard_lookup_ms=hard_lookup_ms,
        nlf_lookup_ms=nlf_lookup_ms,
    )


def _random_bit(example_id, seed):
    value = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).digest()[0]
    return bool(value & 1)


def decide_policies(features: MetaFeatures, *, beta=0.5, threshold=0.0, random_seed=2026):
    """Return frozen D0--D4 and M1--M3a decisions without gold access."""
    hard_score = features.hard.utility
    nlf_score = features.nlf.utility
    hard_route = features.hard.eligible and hard_score > threshold
    nlf_route = features.nlf.eligible and nlf_score > threshold
    hard_v = hard_score - beta * features.hard.standard_error if features.hard.eligible else -math.inf
    nlf_v = nlf_score - beta * features.nlf.standard_error if features.nlf.eligible else -math.inf
    max_utility = max(hard_score, nlf_score)
    max_reliable = max(hard_v, nlf_v)
    selected_u = "hard" if hard_score >= nlf_score else "nlf"
    selected_v = "hard" if hard_v >= nlf_v else "nlf"
    if hard_route == nlf_route:
        random_route = hard_route
    else:
        random_route = hard_route if _random_bit(features.example_id, random_seed) else nlf_route
    # M3a keeps agreement invariant.  In disagreement it accepts the routing
    # side only when its reliability-adjusted utility remains positive.
    if hard_route == nlf_route:
        m3_route = hard_route
        m3_reason = "fixed_policies_agree"
    else:
        routing_side = features.hard if hard_route else features.nlf
        routing_v = routing_side.utility - beta * routing_side.standard_error
        m3_route = routing_side.eligible and routing_v > threshold
        m3_reason = "disagreement_reliability_gate"
    return {
        "D0_HardLabel": PolicyDecision("D0_HardLabel", hard_route, hard_score, "hard", "canonical_hard"),
        "D1_NLF": PolicyDecision("D1_NLF", nlf_route, nlf_score, "nlf", "canonical_nlf"),
        "D2_Union": PolicyDecision("D2_Union", hard_route or nlf_route, max_utility, selected_u, "logical_union"),
        "D3_Intersection": PolicyDecision("D3_Intersection", hard_route and nlf_route, min(hard_score, nlf_score), None, "logical_intersection"),
        "D4_Random": PolicyDecision("D4_Random", random_route, max_utility, None, "seeded_disagreement_choice"),
        "M1_MaxUtility": PolicyDecision("M1_MaxUtility", max_utility > threshold, max_utility, selected_u, "maximum_canonical_utility"),
        "M2_Reliability": PolicyDecision("M2_Reliability", max_reliable > threshold, max_reliable, selected_v, "maximum_reliability_adjusted_utility"),
        "M3a_DisagreementGate": PolicyDecision("M3a_DisagreementGate", m3_route, max_reliable, selected_v, m3_reason),
    }
