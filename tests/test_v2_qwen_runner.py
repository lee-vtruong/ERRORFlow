import json

import pytest

from scripts.run_v2_qwen_baseline import INSTRUCTION_PROFILES, evidence_texts, existing_ids


def test_v2_runner_flattens_structured_evidence():
    row = {"evidence": [{"text": "first"}, "second"]}
    assert evidence_texts(row) == ["first", "second"]


def test_v2_resume_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join([
        json.dumps({"example_id": "dev:0"}),
        json.dumps({"example_id": "dev:0"}),
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate example_id"):
        existing_ids(path)


def test_averitec_instruction_defines_conflicting_and_nei():
    instruction = INSTRUCTION_PROFILES["averitec_v1"]
    assert "CONFLICTING" in instruction
    assert "NOT ENOUGH INFO" in instruction
