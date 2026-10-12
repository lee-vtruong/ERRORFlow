from scripts.evaluate_v2_router_cv import group_name, weighted_metrics


def test_group_name_distinguishes_controls_and_errors():
    assert group_name("A", "A") == "control:A"
    assert group_name("A", "B") == "error:A->B"


def test_weighted_metrics_respect_population_weights():
    accuracy, macro_f1 = weighted_metrics(["A", "B"], ["B", "B"], [9.0, 1.0])
    assert accuracy == 0.1
    assert 0.0 <= macro_f1 <= 1.0
