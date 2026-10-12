from importlib.machinery import SourceFileLoader


def test_fever_retrieval_metrics_measure_any_and_complete_evidence():
    module = SourceFileLoader(
        "evaluate_fever_retrieval", "scripts/evaluate_fever_retrieval.py"
    ).load_module()
    gold = [
        {
            "id": 1,
            "label": "SUPPORTS",
            "evidence": [
                [[0, 0, "Page_A", 0], [0, 1, "Page_A", 1]],
                [[1, 2, "Page_B", 2]],
            ],
        },
        {"id": 2, "label": "NOT ENOUGH INFO", "evidence": []},
    ]
    retrieved = [
        {
            "claim_id": 1,
            "evidence_metadata": [
                {"page_id": "Page_A", "sentence_id": 0},
                {"page_id": "Other", "sentence_id": 0},
                {"page_id": "Page_A", "sentence_id": 1},
            ],
        },
        {"claim_id": 2, "evidence_metadata": []},
    ]
    result = module.retrieval_metrics(gold, retrieved, [1, 3])
    assert result["n_paired_verifiable"] == 1
    assert result["n_paired_without_gold_evidence"] == 1
    assert result["metrics_at_k"]["1"] == {
        "claim_any_sentence_recall": 1.0,
        "complete_evidence_group_recall": 0.0,
        "mean_best_group_coverage": 0.5,
    }
    assert result["metrics_at_k"]["3"]["complete_evidence_group_recall"] == 1.0
