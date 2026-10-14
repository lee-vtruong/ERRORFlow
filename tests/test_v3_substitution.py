import math

from errorflow.v3_memory import ResidualMemoryRecord
from errorflow.v3_substitution import (
    aggregate_similarity,
    bm25_prepare,
    bm25_score,
    estimate_utility,
    label_similarity,
)


def record(recovery=0, regression=0, cost=1.0):
    return ResidualMemoryRecord(
        memory_id=f"m-{recovery}-{regression}", version=1,
        source_dataset="x", source_split="train", source_claim_id="x",
        source_claim="x", error_pattern="x", applicability_conditions=[],
        correction_strategy="a", retrieval_features={}, training_support=1,
        recovery_count=recovery, regression_count=regression,
        estimated_net_utility=float(recovery - regression), estimated_extra_cost=cost,
        validation_status="test",
    )


def test_label_similarity_metrics_are_bounded_and_identical_profiles_match():
    scores = {"SUPPORTED": 2.0, "REFUTED": 0.0, "NOT ENOUGH INFO": -1.0, "CONFLICTING": -2.0}
    for metric in ("l1", "cosine", "l2", "jensen_shannon", "hellinger"):
        value = label_similarity(scores, scores, metric)
        assert 0.999999 <= value <= 1.0


def test_bm25_prefers_overlapping_document():
    counts, idf, average, k1, b = bm25_prepare(["alpha beta", "gamma delta"])
    assert bm25_score("alpha", counts[0], idf, average, k1, b) > bm25_score(
        "alpha", counts[1], idf, average, k1, b
    )


def test_multiplicative_aggregation_penalizes_zero_component():
    weighted = aggregate_similarity((1.0, 1.0, 0.0), mode="weighted_sum")
    multiplied = aggregate_similarity((1.0, 1.0, 0.0), mode="multiplicative")
    assert multiplied < weighted


def test_utility_estimators_preserve_helpful_direction():
    rows = [(record(recovery=1), 0.9), (record(regression=1), 0.1)]
    for mode in ("smoothed", "unsmoothed", "effective_sample_size"):
        assert estimate_utility(rows, mode=mode, cost_weight=0.0)["utility"] > 0
    prior = {"recovery": 0.5, "regression": 0.5, "strength": 2.0}
    assert estimate_utility(rows, mode="bayesian", cost_weight=0.0, prior=prior)["utility"] > 0


def test_zero_similarity_falls_back_to_uniform_weights():
    rows = [(record(recovery=1), 0.0), (record(regression=1), 0.0)]
    result = estimate_utility(rows, mode="unsmoothed", cost_weight=0.0)
    assert math.isclose(result["utility"], 0.0)
    assert result["weight"] == 2.0
