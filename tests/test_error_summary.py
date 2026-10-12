from importlib.machinery import SourceFileLoader

import pytest


def test_error_summary_groups_pairs_confidence_and_sources():
    module = SourceFileLoader("summarize_error_list", "scripts/summarize_error_list.py").load_module()
    rows = [
        {
            "gold_label": "NOT ENOUGH INFO",
            "baseline_prediction": "SUPPORTED",
            "baseline_confidence": 0.83,
            "evidence_metadata": [
                {"source": "exact_claim_title"},
                {"source": "lexical_bm25"},
            ],
        },
        {
            "gold_label": "NOT ENOUGH INFO",
            "baseline_prediction": "SUPPORTED",
            "baseline_confidence": 0.97,
            "evidence_metadata": [{"source": "exact_claim_title"}],
        },
    ]
    result = module.summarize(rows)
    pair = result["label_pairs"]["NOT ENOUGH INFO->SUPPORTED"]
    assert pair["n"] == 2
    assert pair["mean_baseline_confidence"] == pytest.approx(0.9)
    assert result["top1_evidence_source"] == {"exact_claim_title": 2}
