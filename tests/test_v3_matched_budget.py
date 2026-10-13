from scripts.evaluate_v3_matched_budget import evidence_bucket, holm_adjust, mean_interval


def test_evidence_bucket_is_coarse_and_deterministic():
    assert evidence_bucket(0) == "0"
    assert evidence_bucket(1) == "1"
    assert evidence_bucket(2) == "2+"
    assert evidence_bucket(10) == "2+"


def test_holm_adjustment_is_monotonic_in_sorted_order():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.5})
    assert adjusted["a"] == 0.03
    assert adjusted["b"] >= adjusted["a"]
    assert adjusted["c"] >= adjusted["b"]


def test_mean_interval_reports_seed_variation():
    result = mean_interval([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert result["min"] == 1.0
    assert result["max"] == 3.0
