from scripts.analyze_v3_component_significance import fold_report, pair_report


def test_pair_report_direction_is_a0_minus_variant():
    gold = ["A", "B", "A"]
    variant = ["B", "B", "A"]
    a0 = ["A", "B", "A"]
    report = pair_report(gold, variant, a0, samples=100, seed=7)
    assert report["variant_only_correct"] == 0
    assert report["a0_only_correct"] == 1
    assert report["a0_minus_variant_accuracy"] > 0


def test_fold_report_keeps_outer_folds_separate():
    ids = ["a", "b", "c", "d"]
    folds = {"a": 0, "b": 0, "c": 1, "d": 1}
    gold = {"a": "A", "b": "B", "c": "A", "d": "B"}
    variant = {"a": "B", "b": "B", "c": "A", "d": "B"}
    a0 = dict(gold)
    report = fold_report(ids, folds, gold, variant, a0)
    assert len(report["folds"]) == 2
    assert report["folds"][0]["a0_minus_variant_accuracy"] == 0.5
    assert report["folds"][1]["a0_minus_variant_accuracy"] == 0.0
