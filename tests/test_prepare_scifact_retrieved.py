from scripts.prepare_scifact_retrieved import BM25, retrieve


def test_bm25_prefers_matching_document():
    ranker = BM25(["alpha beta", "climate carbon warming"])
    assert ranker.rank("carbon climate", 1)[0][0] == 1


def test_retrieval_does_not_use_cited_or_gold_rationale():
    corpus = [
        {"doc_id": 1, "title": "Alpha", "abstract": ["Unrelated sentence."]},
        {"doc_id": 2, "title": "Climate", "abstract": ["Carbon drives warming.", "Other."]},
    ]
    claims = [{
        "id": 7, "claim": "Carbon drives warming", "cited_doc_ids": [1],
        "evidence": {"1": [{"label": "SUPPORT", "sentences": [0]}]},
    }]
    row = retrieve(corpus, claims, "dev", top_documents=1, top_sentences=1)[0]
    assert row["evidence"][0]["metadata"]["doc_id"] == "2"
    assert row["label"] == "SUPPORTED"
    assert row["metadata"]["retrieval_uses_cited_doc_ids"] is False
