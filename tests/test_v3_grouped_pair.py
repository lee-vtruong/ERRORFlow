import pytest

from scripts.evaluate_v3_grouped_pair import parse_filters


def test_parse_filters_supports_multiple_exact_fields():
    assert parse_filters(["policy=D1_NLF", "mode=natural"]) == {
        "policy": "D1_NLF",
        "mode": "natural",
    }


def test_parse_filters_rejects_missing_separator():
    with pytest.raises(ValueError, match="KEY=VALUE"):
        parse_filters(["natural"])
