from scripts.audit_v3_reproducibility import duplicate_audit, evidence_overlap_audit


def test_duplicate_audit_distinguishes_exact_and_near_cross_fold_pairs():
    rows = [
        {"example_id": "a", "claim": "Alpha beta gamma"},
        {"example_id": "b", "claim": "alpha, beta gamma!"},
        {"example_id": "c", "claim": "Alpha beta gamma delta"},
    ]
    folds = {"a": 0, "b": 0, "c": 1}
    report = duplicate_audit(rows, folds, threshold=0.7)
    assert report["exact_duplicate_groups"] == 1
    assert report["cross_fold_exact_duplicate_groups"] == 0
    assert report["cross_fold_near_duplicate_pairs"] == 2


def test_evidence_overlap_is_descriptive_and_cross_fold():
    rows = [
        {"example_id": "a", "evidence": [{"text": "same evidence", "url": "u"}]},
        {"example_id": "b", "evidence": [{"text": "Same evidence!", "url": "u"}]},
    ]
    report = evidence_overlap_audit(rows, {"a": 0, "b": 1})
    assert report["evidence_fingerprints_shared_across_folds"] == 1
    assert report["source_identifiers_shared_across_folds"] == 1
