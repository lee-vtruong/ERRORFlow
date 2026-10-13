from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap
from scripts.evaluate_v3_significance import INTERPRETATIONS


def test_mcnemar_counts_discordant_pairs_and_returns_valid_p_value():
    result = mcnemar_exact(
        ["A", "A", "B", "B"],
        ["A", "B", "A", "B"],
        ["B", "A", "B", "B"],
    )
    assert result["baseline_only_correct"] == 1
    assert result["candidate_only_correct"] == 2
    assert 0.0 <= result["exact_two_sided_p"] <= 1.0


def test_paired_bootstrap_is_deterministic():
    first = paired_bootstrap(["A", "B"], ["A", "A"], ["A", "B"], samples=20, seed=7)
    second = paired_bootstrap(["A", "B"], ["A", "A"], ["A", "B"], samples=20, seed=7)
    assert first == second


def test_external_significance_interpretation_is_not_train_internal():
    text = INTERPRETATIONS["external-source-only"]
    assert "cross-domain" in text
    assert "source-only memory" in text
    assert "train-internal" not in text
