from errorflow.memory import load_jsonl, save_jsonl
from errorflow.metrics import transition_metrics
from errorflow.router import route
from errorflow.schemas import ErrorMemoryEntry


def test_memory_round_trip(tmp_path):
    entries = [ErrorMemoryEntry("c1", "temporal", "SUPPORTED", best_action="critic", recovery_success=True)]
    path = tmp_path / "memory.jsonl"
    save_jsonl(entries, path)
    assert load_jsonl(path)[0].claim_id == "c1"


def test_router_prefers_empirically_successful_low_cost_action():
    entries = [
        ErrorMemoryEntry("1", "temporal", "SUPPORTED", best_action="critic", recovery_success=True, extra_tokens=100),
        ErrorMemoryEntry("2", "temporal", "SUPPORTED", best_action="retrieve", recovery_success=True, extra_tokens=1000),
    ]
    assert route("temporal", entries).action == "critic"


def test_transition_metrics():
    result = transition_metrics(["A", "B", "A"], ["B", "B", "B"], ["B", "B", "B"])
    assert result["recovery_rate"] == 1.0
    assert result["regression_rate"] == 0.0
