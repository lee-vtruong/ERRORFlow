from errorflow.contracts import ExecutionTrace, PredictionRecord


def test_prediction_contract_is_serializable():
    record = PredictionRecord("c1", "claim", "SUPPORTED", confidence=0.5)
    assert record.to_dict()["claim_id"] == "c1"


def test_trace_defaults_prevent_leakage_flags():
    trace = ExecutionTrace("c1")
    assert trace.official_validation_used is False
    assert trace.test_split_used is False
