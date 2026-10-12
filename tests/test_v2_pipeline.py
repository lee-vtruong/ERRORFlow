from errorflow.error_memory import ErrorMemoryStore, memory_id
from errorflow.pipeline_v2 import InterventionResult, replay_policy
from errorflow.schemas import ErrorMemoryRecord


def record(claim_id):
    return ErrorMemoryRecord(
        memory_id=memory_id("SciFact", claim_id, "overclaim", "verify_scope"),
        dataset="SciFact", source_split="train", claim_id=claim_id,
        claim="drug improves survival in mice", error_type="overclaim",
        baseline_prediction="SUPPORTED", diagnosis="scope mismatch",
        action="verify_scope", recovered=True,
    )


def test_policy_replay_routes_without_mutating_baseline_input():
    memory = ErrorMemoryStore([record("1"), record("2")])
    source = {"claim": "drug improves survival", "dataset": "SciFact", "prediction": "SUPPORTED", "confidence": 0.2}
    result = replay_policy(
        [source], memory,
        {"verify_scope": lambda row, decision: InterventionResult("REFUTED", tokens=12, llm_calls=1)},
        allowed_memory_splits={"train"}, cost_weight=0.0,
    )[0]
    assert result["routed"]
    assert result["final_prediction"] == "REFUTED"
    assert result["llm_calls"] == 2
    assert "baseline_prediction" not in source


def test_policy_replay_falls_back_when_action_not_implemented():
    memory = ErrorMemoryStore([record("1"), record("2")])
    result = replay_policy(
        [{"claim": "drug improves survival", "dataset": "SciFact", "prediction": "SUPPORTED"}],
        memory, {}, allowed_memory_splits={"train"}, cost_weight=0.0,
    )[0]
    assert not result["routed"]
    assert result["final_prediction"] == "SUPPORTED"
