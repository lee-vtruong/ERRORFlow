import pytest

from errorflow.v3_prompts import B1_CONFLICT_AUDIT
from scripts.audit_v3_scifact_adapter import metrics, project
from scripts.run_v3_qwen_action import resolve_protocol


def test_protocol_resolution_keeps_four_label_prompt_immutable():
    labels, prompt = resolve_protocol("conflict_audit_v1", "averitec4")
    assert labels[-1] == "CONFLICTING"
    assert prompt == B1_CONFLICT_AUDIT


def test_scifact_direct_protocol_has_three_labels_and_no_conflicting_verdict():
    labels, prompt = resolve_protocol("conflict_audit_v1", "scifact3")
    assert labels == ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO")
    assert "Choose CONFLICTING" not in prompt
    with pytest.raises(ValueError, match="No frozen SciFact"):
        resolve_protocol("coverage_audit_v2", "scifact3")


def test_projection_and_metrics_make_label_space_effect_explicit():
    gold = ["NOT ENOUGH INFO", "SUPPORTED"]
    raw = ["CONFLICTING", "SUPPORTED"]
    projected = [project(label) for label in raw]
    assert projected == gold
    raw_metrics = metrics(gold, raw, [4, 4])
    projected_metrics = metrics(gold, projected, [4, 4])
    assert raw_metrics["accuracy"] == 0.5
    assert projected_metrics["accuracy"] == 1.0
