from scripts.evaluate_scifact_retrieval import retrieval_metrics


def test_scifact_retrieval_metrics_use_rationale_sentence_ids():
    claims = [{
        "id": 1,
        "evidence": {"10": [{"label": "SUPPORT", "sentences": [1, 2]}]},
    }]
    retrieved = [{
        "example_id": "scifact-dev:1",
        "evidence": [
            {"metadata": {"doc_id": "10", "sentence_id": 1}},
            {"metadata": {"doc_id": "10", "sentence_id": 2}},
        ],
    }]
    result = retrieval_metrics(claims, retrieved, [1, 2])
    assert result["metrics_at_k"]["1"]["claim_any_sentence_recall"] == 1.0
    assert result["metrics_at_k"]["1"]["complete_rationale_recall"] == 0.0
    assert result["metrics_at_k"]["2"]["complete_rationale_recall"] == 1.0
