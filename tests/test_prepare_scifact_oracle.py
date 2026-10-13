from scripts.prepare_scifact_oracle import prepare


def test_prepare_scifact_joins_rationales_and_assigns_nei():
    corpus = [{"doc_id": 10, "title": "Paper", "abstract": ["zero", "evidence sentence"]}]
    claims = [
        {"id": 1, "claim": "claim", "evidence": {"10": [{"label": "SUPPORT", "sentences": [1]}]}},
        {"id": 2, "claim": "unknown", "evidence": {}},
    ]
    rows, diagnostics = prepare(corpus, claims, "dev")
    assert rows[0]["example_id"] == "scifact-dev:1"
    assert rows[0]["label"] == "SUPPORTED"
    assert rows[0]["evidence"][0]["text"] == "evidence sentence"
    assert rows[1]["label"] == "NOT ENOUGH INFO"
    assert diagnostics == {"missing_documents": 0, "invalid_sentence_indices": 0}
