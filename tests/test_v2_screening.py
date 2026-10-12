from scripts.build_v2_screening_set import build_screen
from scripts.evaluate_v2_intervention import evaluate


def test_screen_is_stratified_and_deterministic():
    gold = [
        {"example_id": "train:0", "label": "A", "claim": "x"},
        {"example_id": "train:1", "label": "A", "claim": "y"},
        {"example_id": "train:2", "label": "B", "claim": "z"},
    ]
    baseline = [
        {"example_id": "train:0", "prediction": "B"},
        {"example_id": "train:1", "prediction": "A"},
        {"example_id": "train:2", "prediction": "B"},
    ]
    first, counts = build_screen(gold, baseline, 1, 1, 3)
    second, _ = build_screen(gold, baseline, 1, 1, 3)
    assert [row["example_id"] for row in first] == [row["example_id"] for row in second]
    assert counts["errors"] == {"A->B": 1}
    assert sum(counts["controls"].values()) == 2


def test_intervention_evaluation_counts_recovery_and_regression():
    gold = [{"example_id": "1", "label": "A"}, {"example_id": "2", "label": "B"}]
    baseline = [{"example_id": "1", "prediction": "B"}, {"example_id": "2", "prediction": "B"}]
    candidate = [{"example_id": "1", "prediction": "A"}, {"example_id": "2", "prediction": "A"}]
    result = evaluate(gold, baseline, candidate)
    assert result["helpful"] == 1
    assert result["harmful"] == 1
    assert result["net_correct_change"] == 0
