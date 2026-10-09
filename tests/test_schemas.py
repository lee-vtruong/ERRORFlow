from errorflow.schemas import ErrorMemoryEntry


def test_error_memory_entry_defaults_are_explicit():
    entry = ErrorMemoryEntry(
        claim_id="example_001",
        error_type="temporal_mismatch",
        baseline_prediction="SUPPORTED",
    )
    assert entry.gold_label is None
    assert entry.candidate_actions == []
    assert entry.recovery_success is None
