from scripts.build_v3_action_screen import build_screen, spread_sample


def test_spread_sample_covers_low_and_high_margin():
    rows = [{"example_id": str(index), "prediction_margin": index / 10} for index in range(10)]
    selected = spread_sample(rows, 3, 2026, "group")
    assert [row["prediction_margin"] for row in selected] == [0.0, 0.4, 0.9]


def test_screen_contains_errors_and_correct_controls():
    gold, baseline = [], []
    for index, (expected, predicted) in enumerate([
        ("SUPPORTED", "SUPPORTED"), ("SUPPORTED", "REFUTED"),
        ("REFUTED", "REFUTED"), ("REFUTED", "SUPPORTED"),
    ]):
        key = f"train:{index}"
        gold.append({"example_id": key, "dataset": "AVeriTeC", "split": "train", "claim": key,
                     "label": expected, "evidence": []})
        baseline.append({"example_id": key, "claim": key, "gold_label": expected, "prediction": predicted,
                         "correct": expected == predicted, "prediction_margin": 0.1 * index})
    examples, selected, manifest = build_screen(gold, baseline, 1, 1)
    assert len(examples) == len(selected) == 4
    assert set(manifest["groups"]) == {
        "control:REFUTED", "control:SUPPORTED", "error:REFUTED->SUPPORTED", "error:SUPPORTED->REFUTED",
    }
