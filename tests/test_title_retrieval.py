from importlib.machinery import SourceFileLoader


def test_claim_and_fever_page_title_keys_match():
    module = SourceFileLoader("retrieve_fever_titles", "scripts/retrieve_fever_titles.py").load_module()
    claim_keys = module.claim_title_keys("Nikolaj Coster-Waldau worked with Fox.")
    page_keys = module.page_title_keys("Nikolaj_Coster-Waldau")
    assert claim_keys & page_keys


def test_disambiguated_page_also_exposes_base_title():
    module = SourceFileLoader("retrieve_fever_titles_base", "scripts/retrieve_fever_titles.py").load_module()
    assert "mercury" in module.page_title_keys("Mercury_-LRB-planet-RRB-")


def test_hybrid_merge_deduplicates_and_preserves_provenance():
    module = SourceFileLoader("merge_retrieval_candidates", "scripts/merge_retrieval_candidates.py").load_module()
    lexical = {
        "claim_id": "1",
        "evidence": ["lexical duplicate", "lexical second"],
        "evidence_metadata": [
            {"page_id": "A", "sentence_id": 0, "rank": 1},
            {"page_id": "B", "sentence_id": 0, "rank": 2},
        ],
    }
    title = {
        "claim_id": "1",
        "evidence": ["title A", "title C"],
        "evidence_metadata": [
            {"page_id": "A", "sentence_id": 0, "rank": 1, "source": "title"},
            {"page_id": "C", "sentence_id": 1, "rank": 2, "source": "title"},
        ],
    }
    merged = module.merge_row(lexical, title, k=3, title_first=1)
    assert [(m["page_id"], m["sentence_id"]) for m in merged["evidence_metadata"]] == [
        ("A", 0), ("B", 0), ("C", 1)
    ]
