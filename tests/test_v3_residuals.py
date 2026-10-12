from errorflow.v3_residuals import build_residual_records, summarize_residuals


def gold(example_id, label, claim="claim"):
    return {"example_id": example_id, "dataset": "AVeriTeC", "split": "train", "claim": claim,
            "label": label, "evidence": [{"evidence_id": "e1", "text": "some evidence", "source": "web"}]}


def prediction(example_id, label, scores):
    return {"example_id": example_id, "prediction": label, "latency_ms": 5, "token_count": 0,
            "metadata": {"label_scores": scores, "model_forward_passes": 4,
                         "instruction_profile": "conflict_audit_v1", "decoding_mode": "label-score"}}


def test_residual_analysis_separates_transition_from_causal_status():
    rows = build_residual_records(
        [gold("train:1", "NOT ENOUGH INFO")],
        [prediction("train:1", "REFUTED", {"REFUTED": -1, "SUPPORTED": -3, "NOT ENOUGH INFO": -2})],
    )
    assert rows[0]["error_transition"] == "NOT ENOUGH INFO->REFUTED"
    assert rows[0]["diagnosis"]["error_type"] == "insufficient_evidence_mistaken_as_refutation"
    assert rows[0]["diagnosis"]["status"] == "hypothesis_unvalidated"
    summary = summarize_residuals(rows)
    assert summary["n_errors"] == 1
    assert summary["causal_diagnosis_status"] == "not_validated"
