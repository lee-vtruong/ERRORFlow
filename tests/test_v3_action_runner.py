import json

import pytest

from errorflow.io import config_hash
from scripts.run_v3_qwen_action import (
    ACTION_PROMPTS, sample_cache_key, sha256_text, validate_cached_rows,
)


def row():
    return {"example_id": "train:1", "claim": "claim", "evidence": [{"evidence_id": "e1", "text": "evidence"}]}


def test_cache_key_changes_with_prompt_and_validates_cached_output():
    generation_hash = config_hash({"decoding": "score"}, length=32)
    model_hash = sha256_text("model")
    prompt_hash = sha256_text(ACTION_PROMPTS["conflict_audit_v1"])
    key, evidence_hash = sample_cache_key(row(), model_hash, prompt_hash, generation_hash)
    cached = {"train:1": {"example_id": "train:1", "metadata": {"cache_key": key}}}
    validate_cached_rows(cached, [row()], model_hash, prompt_hash, generation_hash)
    assert evidence_hash
    with pytest.raises(ValueError, match="Cache key mismatch"):
        validate_cached_rows(cached, [row()], model_hash, sha256_text("changed"), generation_hash)


def test_all_v3_actions_are_exposed_by_runner():
    assert "conflict_audit_v1" in ACTION_PROMPTS
    assert "structured_decomposition_v1" in ACTION_PROMPTS
