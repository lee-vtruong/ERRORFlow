import pytest

from scripts.plot_v3_external_results import POLICIES, validate_summary


def test_external_plot_summary_requires_all_policies():
    summary = {"results": {policy: {} for policy in POLICIES}}
    assert validate_summary(summary, "test") is summary
    del summary["results"]["R6"]
    with pytest.raises(ValueError, match="R6"):
        validate_summary(summary, "test")
