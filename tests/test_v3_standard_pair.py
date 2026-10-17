import json

import pytest

from errorflow.io import write_jsonl
from errorflow.v3_prompts import V3_ACTION_PROMPTS, prompt_sha256
from scripts.evaluate_v3_standard_pair import (
    ALT_ACTION,
    BASE_ACTION,
    load_complete_manifest,
    validate_pair,
)


LABELS = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO", "CONFLICTING"]


def write_predictions(tmp_path, stem, action, ids=("x:1", "x:2")):
    path = tmp_path / f"{stem}.jsonl"
    prompt_hash = prompt_sha256(V3_ACTION_PROMPTS[action]) if action in V3_ACTION_PROMPTS else "audit"
    rows = [{
        "example_id": key,
        "prediction": "SUPPORTED",
        "metadata": {
            "instruction_profile": action,
            "prompt_sha256": prompt_hash,
            "gold_label_accessed": False,
        },
    } for key in ids]
    write_jsonl(path, rows)
    manifest = {
        "config_id": f"config-{stem}",
        "status": "complete",
        "requested_rows": len(rows),
        "completed_rows": len(rows),
        "config": {
            "action": action,
            "prompt_sha256": prompt_hash,
            "model": "frozen-model",
            "model_hash": "model-hash",
            "input": "gold.jsonl",
            "input_sha256": "input-hash",
            "split": "train",
            "seed": 2026,
            "generation_config_hash": "generation-hash",
            "generation": {"labels": LABELS, "forward_passes": 4},
        },
    }
    path.with_name(path.stem + ".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    return path


def test_standard_pair_accepts_only_matching_frozen_pair(tmp_path):
    base = write_predictions(tmp_path, "base", BASE_ACTION)
    alt = write_predictions(tmp_path, "alt", ALT_ACTION)
    result = validate_pair(base, alt)
    assert result["status"] == "validated"
    assert result["historical_audit_memory_allowed"] is False
    assert result["rows"] == 2


def test_standard_pair_rejects_historical_audit_action(tmp_path):
    audit = write_predictions(tmp_path, "audit", "conflict_audit_v1")
    with pytest.raises(ValueError, match="forbidden"):
        load_complete_manifest(audit, BASE_ACTION)


def test_standard_pair_rejects_model_mismatch(tmp_path):
    base = write_predictions(tmp_path, "base", BASE_ACTION)
    alt = write_predictions(tmp_path, "alt", ALT_ACTION)
    manifest_path = alt.with_name("alt.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["model_hash"] = "different"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest mismatch"):
        validate_pair(base, alt)
