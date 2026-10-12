from scripts.evaluate_v3_fold_stability import summarize_folds


def test_fold_stability_reports_mixed_fold_directions():
    baseline = [
        {"example_id": "a", "outer_fold": 0, "gold_label": "A", "prediction": "B"},
        {"example_id": "b", "outer_fold": 0, "gold_label": "B", "prediction": "B"},
        {"example_id": "c", "outer_fold": 1, "gold_label": "A", "prediction": "A"},
        {"example_id": "d", "outer_fold": 1, "gold_label": "B", "prediction": "B"},
    ]
    candidate = [
        {"example_id": "a", "prediction": "A", "routed": True},
        {"example_id": "b", "prediction": "B", "routed": False},
        {"example_id": "c", "prediction": "B", "routed": True},
        {"example_id": "d", "prediction": "B", "routed": False},
    ]
    report = summarize_folds(baseline, candidate)
    assert report["n_folds"] == 2
    assert report["positive_accuracy_folds"] == 1
    assert report["folds"][0]["net"] == 1
    assert report["folds"][1]["net"] == -1
