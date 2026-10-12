from scripts.build_v2_action_memory import build_records


def test_action_memory_keeps_recovery_and_regression():
    gold = [
        {"example_id": "train:0", "claim": "x", "label": "A", "evidence": []},
        {"example_id": "train:1", "claim": "y", "label": "B", "evidence": []},
    ]
    baseline = [
        {"example_id": "train:0", "prediction": "B", "metadata": {"label_scores": {"A": 0.0}}},
        {"example_id": "train:1", "prediction": "B", "metadata": {"label_scores": {"B": 1.0}}},
    ]
    candidate = [
        {"example_id": "train:0", "prediction": "A", "metadata": {}},
        {"example_id": "train:1", "prediction": "A", "metadata": {}},
    ]
    records = build_records(gold, baseline, [("critic", candidate)])
    assert records[0].utility == 1.0
    assert records[1].utility == -1.0
    assert records[1].metadata["record_role"] == "control"
