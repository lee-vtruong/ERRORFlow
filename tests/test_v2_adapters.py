import json

import pytest

from errorflow.adapters import get_adapter


def test_averitec_adapter_preserves_qa_and_url():
    row = {
        "claim_id": 7,
        "claim": "A claim",
        "label": "Supported",
        "questions": [{"question": "Why?", "answers": [{"answer": "Because.", "url": "https://example.test"}]}],
    }
    example = get_adapter("averitec").normalize(row, "train")
    assert example.example_id == "7"
    assert example.label == "SUPPORTED"
    assert example.evidence[0].url == "https://example.test"


def test_scifact_adapter_normalizes_labels():
    example = get_adapter("scifact").normalize(
        {"id": 9, "claim": "Scientific claim", "label": "CONTRADICT", "evidence_text": ["Abstract sentence"]},
        "dev",
    )
    assert example.label == "REFUTED"
    assert example.evidence[0].source == "scientific_abstract"


def test_unknown_adapter_is_explicit():
    with pytest.raises(ValueError, match="Unknown dataset adapter"):
        get_adapter("unknown")


def test_averitec_not_enough_evidence_label_variants():
    adapter = get_adapter("averitec")
    for label in ("Not Enough Evidence", "NOT_ENOUGH_EVIDENCE"):
        assert adapter.normalize({"id": 1, "claim": "x", "label": label}, "dev").label == "NOT ENOUGH INFO"


def test_adapter_reads_official_json_array_and_source_url(tmp_path):
    path = tmp_path / "dev.json"
    path.write_text(json.dumps([{"claim": "x", "label": "Supported",
        "questions": [{"question": "q", "answers": [{"answer": "a", "source_url": "https://source.test"}]}]}]), encoding="utf-8")
    example = list(get_adapter("averitec").iter_file(path, "dev"))[0]
    assert example.evidence[0].url == "https://source.test"
