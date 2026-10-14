from scripts.export_v3_worked_example import CASE_ORDER, case_kind


def test_case_kind_finds_harm_prevention_and_helpful_rescue():
    assert case_kind("SUPPORTED", "SUPPORTED", "REFUTED", False, True) == CASE_ORDER[0]
    assert case_kind("SUPPORTED", "REFUTED", "SUPPORTED", True, False) == CASE_ORDER[1]


def test_case_kind_rejects_non_causal_route_patterns():
    assert case_kind("SUPPORTED", "SUPPORTED", "REFUTED", True, True) is None
    assert case_kind("SUPPORTED", "REFUTED", "SUPPORTED", True, True) is None
    assert case_kind("SUPPORTED", "SUPPORTED", "SUPPORTED", False, True) is None
