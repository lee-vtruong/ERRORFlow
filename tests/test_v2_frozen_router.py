from errorflow.error_memory import ErrorMemoryStore, memory_id
from errorflow.schemas import ErrorMemoryRecord
from scripts.run_v2_frozen_router import replay


def test_frozen_router_replay_uses_candidate_without_reading_gold():
    action = "audit"
    memories = [
        ErrorMemoryRecord(
            memory_id=memory_id("AVeriTeC", str(index), "error", action),
            dataset="AVeriTeC", source_split="train", claim_id=str(index),
            claim="same claim", error_type="error", baseline_prediction="REFUTED",
            diagnosis="error", action=action, recovered=True, utility=1.0,
            metadata={"model_forward_passes": 4},
        ) for index in range(2)
    ]
    examples = [{"example_id": "dev:1", "dataset": "AVeriTeC", "split": "dev", "claim": "same claim", "label": "SUPPORTED"}]
    baseline = {"dev:1": {"example_id": "dev:1", "prediction": "REFUTED", "confidence": 0.1, "metadata": {"model_forward_passes": 4}}}
    candidates = {action: {"dev:1": {"example_id": "dev:1", "prediction": "SUPPORTED", "metadata": {"model_forward_passes": 4}}}}
    config = {
        "policy_id": "test", "actions": [action], "memory_splits": ["train"],
        "top_k_per_action": 2, "min_support": 2, "min_similarity": 0.0,
        "min_utility": 0.0, "cost_weight": 0.0, "uncertainty_mode": "inverse_confidence",
    }
    result = replay(examples, baseline, candidates, ErrorMemoryStore(memories), config)[0]
    assert result["prediction"] == "SUPPORTED"
    assert result["routed"] is True
    assert result["metadata"]["model_forward_passes"] == 8
    assert "label" not in result
