import pytest
from dataclasses import replace

from errorflow.v3_memory import ResidualMemoryStore, build_action_memories, smoothed_utility
from errorflow.v3_routing import best_memory_action, select_budget, select_weighted_budget


def residual(example_id="train:1", correct=False):
    return {
        "example_id": example_id, "dataset": "AVeriTeC", "split": "train", "claim": "alpha beta",
        "gold_label": "SUPPORTED", "prediction": "REFUTED", "correct": correct,
        "diagnosis": {"error_type": "false_contradiction"}, "label_scores": {"REFUTED": -1, "SUPPORTED": -2},
        "prediction_margin": 0.2, "claim_tokens": 2, "evidence_count": 1, "evidence_tokens": 5, "outer_fold": 0,
    }


def candidate(example_id="train:1", prediction="SUPPORTED"):
    return {"example_id": example_id, "prediction": prediction, "latency_ms": 10, "token_count": 0,
            "metadata": {"model_forward_passes": 4}}


def test_memory_build_retrieve_deactivate_and_fold_exclusion(tmp_path):
    records = build_action_memories([residual()], [("audit", [candidate()])])
    assert records[0].recovery_count == 1
    store = ResidualMemoryStore(records)
    query = {"claim": "alpha", "retrieval_features": {
        "baseline_prediction": "REFUTED", "label_scores": {"REFUTED": -1, "SUPPORTED": -2}, "evidence_tokens": 5,
    }}
    assert len(store.retrieve(query)) == 1
    assert store.retrieve(query, exclude_folds={0}) == []
    store.deactivate(records[0].memory_id, "test")
    assert store.retrieve(query) == []
    store.upsert(replace(records[0], version=2, active=True))
    with pytest.raises(ValueError, match="older version"):
        store.upsert(records[0])
    path = tmp_path / "memory.jsonl"
    store.save(path)
    assert len(ResidualMemoryStore.load(path)) == 1


def test_memory_rejects_non_train_and_utility_is_smoothed():
    row = residual("dev:25")
    row["split"] = "dev"
    with pytest.raises(ValueError, match="Only AVeriTeC train"):
        build_action_memories([row], [("audit", [candidate("dev:25")])])
    assert smoothed_utility(3, 1, 5) > 0


def test_memory_records_harm_on_correct_controls():
    row = residual(correct=True)
    row["prediction"] = "SUPPORTED"
    records = build_action_memories([row], [("audit", [candidate(prediction="REFUTED")])])
    assert records[0].error_pattern == "correct_control"
    assert records[0].recovery_count == 0
    assert records[0].regression_count == 1
    assert records[0].estimated_net_utility == -1.0


def test_memory_uses_four_pass_cost_for_legacy_rows_without_metadata():
    legacy = candidate()
    legacy["metadata"] = {}
    records = build_action_memories([residual()], [("audit", [legacy])])
    assert records[0].estimated_extra_cost == 1.0


def test_utility_router_and_budget_are_deterministic():
    records = build_action_memories(
        [residual("train:1"), residual("train:2")],
        [("audit", [candidate("train:1"), candidate("train:2")])],
    )
    neighbors = [(record, 1.0) for record in records]
    best, estimates = best_memory_action(["audit"], neighbors, cost_weight=0.1, min_support=2)
    assert best.action == "audit"
    assert best.utility > 0
    assert select_budget(["a", "b", "c", "d"], {"a": 4, "b": 3, "c": 2, "d": 1}, 0.5) == {"a", "b"}
    assert select_weighted_budget(["a", "b"], {"a": 2, "b": 1}, {"a": 1, "b": 9}, 0.1) == {"a"}


def test_memory_size_ranking_cache_is_invalidated_on_update():
    records = build_action_memories([residual()], [("audit", [candidate()])])
    store = ResidualMemoryStore(records)
    query = {"claim": "alpha", "retrieval_features": {
        "baseline_prediction": "REFUTED", "label_scores": {"REFUTED": -1, "SUPPORTED": -2},
        "evidence_tokens": 5,
    }}
    assert len(store.retrieve(query, max_records=1)) == 1
    second = build_action_memories(
        [residual("train:2")], [("audit", [candidate("train:2")])]
    )[0]
    store.upsert(second)
    assert len(store.retrieve(query, max_records=2)) == 2
