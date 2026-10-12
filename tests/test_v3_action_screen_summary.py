import pytest

from scripts.summarize_v3_action_screen import summarize_screen


def record(key, gold, prediction, correct):
    return {
        "example_id": key, "gold_label": gold, "prediction": prediction,
        "correct": correct, "claim": key,
    }


def test_summary_counts_help_and_harm_with_population_weights():
    population = [
        record("p1", "SUPPORTED", "SUPPORTED", True),
        record("p2", "SUPPORTED", "SUPPORTED", True),
        record("p3", "REFUTED", "SUPPORTED", False),
        record("p4", "REFUTED", "SUPPORTED", False),
        record("p5", "REFUTED", "SUPPORTED", False),
    ]
    screen = [population[0], population[2]]
    candidate = [
        {"example_id": "p1", "prediction": "REFUTED"},
        {"example_id": "p3", "prediction": "REFUTED"},
    ]
    report = summarize_screen(screen, population, [("action", candidate)])
    action = report["actions"]["action"]
    assert action["helpful"] == 1
    assert action["harmful"] == 1
    assert action["weighted_helpful"] == 3
    assert action["weighted_harmful"] == 2
    assert action["weighted_net"] == 1


def test_summary_rejects_candidate_id_mismatch():
    row = record("p1", "SUPPORTED", "SUPPORTED", True)
    with pytest.raises(ValueError, match="Candidate IDs differ"):
        summarize_screen([row], [row], [("action", [])])
