import pytest

from errorflow.v3_folds import assign_grouped_folds, reject_opened_holdout, validate_fold_isolation
from errorflow.v3_prompts import B1_CONFLICT_AUDIT, B1_CONFLICT_AUDIT_SHA256, V3_ACTION_PROMPTS, prompt_sha256
from scripts.run_v2_qwen_baseline import INSTRUCTION_PROFILES


def test_b1_prompt_is_byte_identical_to_historical_runner():
    assert prompt_sha256(B1_CONFLICT_AUDIT) == B1_CONFLICT_AUDIT_SHA256
    assert INSTRUCTION_PROFILES["conflict_audit_v1"] == B1_CONFLICT_AUDIT
    assert set(V3_ACTION_PROMPTS) == {
        "coverage_audit_v2", "refutation_audit_v2", "conflicting_rescue_v1",
        "entity_scope_audit_v1", "structured_decomposition_v1",
    }


def test_grouped_folds_keep_duplicate_claims_together():
    rows = [
        {"example_id": "a", "claim": "The same claim!"},
        {"example_id": "b", "claim": "the same claim"},
        {"example_id": "c", "claim": "different"},
    ]
    folds = assign_grouped_folds(rows, n_folds=2)
    assert folds["a"] == folds["b"]


def test_opened_holdout_and_fold_overlap_are_rejected():
    with pytest.raises(ValueError, match="opened AVeriTeC holdout"):
        reject_opened_holdout([{"example_id": "dev:25"}])
    with pytest.raises(ValueError, match="Fold contamination"):
        validate_fold_isolation({"train:1"}, {"train:1"})
