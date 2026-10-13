from scripts.evaluate_v3_external_transfer import build_routes, evaluate
from errorflow.v3_memory import ResidualMemoryRecord, ResidualMemoryStore


def memory(claim_id, utility):
    return ResidualMemoryRecord(
        memory_id=f"m-{claim_id}", version=1, source_dataset="AVeriTeC",
        source_split="train", source_claim_id=claim_id, source_claim="climate claim",
        error_pattern="x", applicability_conditions=[], correction_strategy="action",
        retrieval_features={"baseline_prediction": "REFUTED", "label_scores": {"REFUTED": 1.0}},
        training_support=1, recovery_count=int(utility > 0), regression_count=int(utility < 0),
        estimated_net_utility=utility, estimated_extra_cost=1.0,
        validation_status="test",
    )


def row(example_id, prediction="REFUTED"):
    return {
        "example_id": example_id, "claim": "climate claim", "prediction": prediction,
        "confidence": 0.6,
        "metadata": {"label_scores": {"REFUTED": 1.0}, "evidence_tokens": 10},
    }


def test_external_routes_match_control_counts_without_gold():
    ids = ["a", "b"]
    baseline = {key: row(key) for key in ids}
    store = ResidualMemoryStore([memory("s1", 1), memory("s2", 1)])
    routes, _ = build_routes(ids, baseline, store, "action", 2, 2, 0.1, 0.0)
    assert len(routes["R-Confidence"]) == len(routes["R6"])
    assert len(routes["R-Similarity"]) == len(routes["R6"])


def test_evaluation_accounts_for_helpful_and_harmful_flips():
    ids = ["a", "b"]
    gold = {"a": {"label": "SUPPORTED"}, "b": {"label": "REFUTED"}}
    baseline = {"a": row("a"), "b": row("b")}
    candidate = {"a": row("a", "SUPPORTED"), "b": row("b", "SUPPORTED")}
    result, _ = evaluate(ids, gold, baseline, candidate, set(ids))
    assert result["helpful"] == 1
    assert result["harmful"] == 1
    assert result["net"] == 0


def test_evaluation_projects_only_metric_predictions():
    ids = ["a"]
    gold = {"a": {"label": "NOT ENOUGH INFO"}}
    baseline = {"a": row("a", "CONFLICTING")}
    candidate = {"a": row("a", "SUPPORTED")}
    result, predictions = evaluate(
        ids, gold, baseline, candidate, set(), {"CONFLICTING": "NOT ENOUGH INFO"}
    )
    assert predictions == ["NOT ENOUGH INFO"]
    assert result["accuracy"] == 1.0
