from importlib.machinery import SourceFileLoader


def test_rerank_row_sorts_scores_and_keeps_old_rank():
    module = SourceFileLoader("rerank_cross_encoder", "scripts/rerank_cross_encoder.py").load_module()
    row = {
        "claim_id": "1",
        "evidence": ["first", "second"],
        "evidence_metadata": [
            {"page_id": "A", "sentence_id": 0, "rank": 1},
            {"page_id": "B", "sentence_id": 1, "rank": 2},
        ],
        "retrieval": {"method": "union"},
    }
    result = module.rerank_row(row, [-1.0, 2.0], "test-model")
    assert result["evidence"] == ["second", "first"]
    assert result["evidence_metadata"][0]["pre_rerank_rank"] == 2
    assert result["evidence_metadata"][0]["reranker_score"] == 2.0


def test_select_topk_keeps_evidence_aligned():
    module = SourceFileLoader("select_retrieval_topk", "scripts/select_retrieval_topk.py").load_module()
    row = {
        "claim_id": "1",
        "evidence": ["one", "two"],
        "evidence_metadata": [
            {"page_id": "A", "sentence_id": 0},
            {"page_id": "B", "sentence_id": 1},
        ],
    }
    result = module.select_topk(row, 1)
    assert result["evidence"] == ["one"]
    assert result["evidence_metadata"] == [{"page_id": "A", "sentence_id": 0, "rank": 1}]
