import json

from scripts.download_climate_fever import download, parse_and_audit


def example(claim_id="1", label="DISPUTED"):
    return {
        "claim_id": claim_id, "claim": "A climate claim", "claim_label": label,
        "evidences": [{"evidence_label": "SUPPORTS", "evidence": "text"}] * 5,
    }


def test_parse_and_audit_counts_claims_labels_and_evidence():
    payload = (json.dumps(example()) + "\n" + json.dumps(example("2", "REFUTES")) + "\n").encode()
    _, report = parse_and_audit(payload)
    assert report["rows"] == 2
    assert report["claim_labels"] == {"DISPUTED": 1, "REFUTES": 1}
    assert report["evidence_count_distribution"] == {"5": 2}


def test_download_reuses_existing_audited_file(tmp_path):
    target = tmp_path / "climate-fever.jsonl"
    target.write_text(json.dumps(example()) + "\n", encoding="utf-8")
    result = download(target)
    assert result["status"] == "reused"
    assert result["rows"] == 1
