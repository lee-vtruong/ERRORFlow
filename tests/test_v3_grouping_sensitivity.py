from scripts.evaluate_v3_grouping_sensitivity import grouped_folds, near_duplicate_edges


def test_near_duplicate_components_stay_in_same_fold():
    rows = [
        {"example_id": "a", "claim": "alpha beta gamma delta"},
        {"example_id": "b", "claim": "alpha beta gamma"},
        {"example_id": "c", "claim": "unrelated example here"},
        {"example_id": "d", "claim": "another separate claim"},
    ]
    edges = near_duplicate_edges(rows, 0.7)
    assignments, components = grouped_folds(rows, edges, 0.7, n_folds=2, seed=7)
    assert assignments["a"] == assignments["b"]
    assert any(set(component) == {"a", "b"} for component in components)


def test_threshold_controls_component_merging():
    rows = [
        {"example_id": "a", "claim": "one two three four"},
        {"example_id": "b", "claim": "one two three"},
        {"example_id": "c", "claim": "five six seven"},
        {"example_id": "d", "claim": "eight nine ten"},
    ]
    edges = near_duplicate_edges(rows, 0.7)
    low, low_components = grouped_folds(rows, edges, 0.7, n_folds=2, seed=7)
    _, high_components = grouped_folds(rows, edges, 0.9, n_folds=2, seed=7)
    assert low["a"] == low["b"]
    assert any(set(component) == {"a", "b"} for component in low_components)
    assert not any(set(component) == {"a", "b"} for component in high_components)
