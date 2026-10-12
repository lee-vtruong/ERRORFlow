from errorflow.evaluation_v2 import evaluate_run
from errorflow.io import config_hash, read_jsonl, write_jsonl


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_atomic_jsonl_round_trip(tmp_path):
    path = tmp_path / "nested" / "rows.jsonl"
    write_jsonl(path, [{"example_id": "1"}, {"example_id": "2"}])
    assert list(read_jsonl(path))[-1]["example_id"] == "2"
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_evaluate_run_tracks_helpful_harmful_and_cost():
    rows = [
        {"gold_label": "S", "baseline_prediction": "R", "final_prediction": "S", "routed": True, "llm_calls": 2},
        {"gold_label": "S", "baseline_prediction": "S", "final_prediction": "R", "routed": True, "llm_calls": 2},
        {"gold_label": "R", "baseline_prediction": "R", "routed": False, "llm_calls": 1},
    ]
    result = evaluate_run(rows)
    assert result["helpful"] == 1
    assert result["harmful"] == 1
    assert result["net_recovery"] == 0
    assert result["route_rate"] == 2 / 3
