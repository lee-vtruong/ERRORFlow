import pytest

from errorflow.datasets import normalize_fever_label


def test_fever_labels_match_verifier_labels():
    assert normalize_fever_label("SUPPORTS") == "SUPPORTED"
    assert normalize_fever_label("REFUTES") == "REFUTED"
    assert normalize_fever_label("NOT ENOUGH INFO") == "NOT ENOUGH INFO"


def test_unknown_fever_label_fails_closed():
    with pytest.raises(ValueError):
        normalize_fever_label("UNKNOWN")
