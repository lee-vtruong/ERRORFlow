from errorflow.error_memory import ErrorMemoryStore, memory_id
from errorflow.routing_v2 import select_action
from errorflow.schemas import ErrorMemoryRecord


def make_record(claim_id, action, recovered, split="train", allowed=True, calls=1):
    return ErrorMemoryRecord(
        memory_id=memory_id("AVeriTeC", claim_id, "missing_evidence", action),
        dataset="AVeriTeC",
        source_split=split,
        claim_id=claim_id,
        claim="The city opened a new hospital in 2020",
        error_type="missing_evidence",
        baseline_prediction="SUPPORTED",
        diagnosis="Evidence is incomplete",
        action=action,
        recovered=recovered,
        extra_llm_calls=calls,
        allow_for_routing=allowed,
    )


def test_store_upserts_and_round_trips(tmp_path):
    first = make_record("1", "retrieve_more", True)
    replacement = make_record("1", "retrieve_more", False)
    store = ErrorMemoryStore([first, replacement])
    path = tmp_path / "memory.jsonl"
    store.save(path)
    loaded = ErrorMemoryStore.load(path)
    assert len(loaded) == 1
    assert loaded.retrieve("city hospital 2020", dataset="AVeriTeC")[0][0].recovered is False


def test_retrieval_enforces_split_and_routing_guard():
    store = ErrorMemoryStore([
        make_record("train", "retrieve_more", True, split="train"),
        make_record("test", "retrieve_more", True, split="test"),
        make_record("blocked", "retrieve_more", True, allowed=False),
    ])
    found = store.retrieve("city hospital", dataset="AVeriTeC", allowed_splits={"train"})
    assert [record.claim_id for record, _ in found] == ["train"]


def test_router_uses_recovery_uncertainty_and_cost():
    rows = [
        (make_record("1", "cheap", True, calls=1), 0.8),
        (make_record("2", "cheap", True, calls=1), 0.7),
        (make_record("3", "expensive", True, calls=5), 0.9),
        (make_record("4", "expensive", True, calls=5), 0.9),
    ]
    decision = select_action(rows, uncertainty=0.8, cost_weight=0.1)
    assert decision.route
    assert decision.action == "cheap"


def test_router_abstains_without_support():
    decision = select_action([(make_record("1", "critic", True), 0.8)], uncertainty=1.0, min_support=2)
    assert not decision.route
    assert decision.action == "none"


def test_retrieval_conditions_on_prediction_and_score_profile():
    supported = make_record("s", "critic", True)
    supported.baseline_prediction = "SUPPORTED"
    supported.metadata["baseline_label_scores"] = {"SUPPORTED": 3.0, "REFUTED": 0.0}
    refuted = make_record("r", "critic", True)
    refuted.baseline_prediction = "REFUTED"
    refuted.metadata["baseline_label_scores"] = {"SUPPORTED": 0.0, "REFUTED": 3.0}
    store = ErrorMemoryStore([supported, refuted])
    found = store.retrieve(
        "unrelated query", dataset="AVeriTeC", baseline_prediction="SUPPORTED",
        label_scores={"SUPPORTED": 2.5, "REFUTED": 0.0},
    )
    assert [record.claim_id for record, _ in found] == ["s"]


def test_router_uses_negative_control_utility():
    harmful = make_record("h1", "critic", False)
    harmful.utility = -1.0
    harmful2 = make_record("h2", "critic", False)
    harmful2.utility = -1.0
    decision = select_action([(harmful, 0.9), (harmful2, 0.8)], uncertainty=1.0, cost_weight=0.0)
    assert not decision.route
