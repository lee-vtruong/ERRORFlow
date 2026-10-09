from errorflow.fever_evidence import evidence_refs, resolve_sentence


def test_extracts_fever_evidence_reference():
    row = {"evidence": [[[0, 1, "Example_Page", 2]]]}
    assert evidence_refs(row) == [("Example_Page", 2)]


def test_resolves_sentence_from_page_lines():
    page = {"lines": "0\tFirst.\n2\tSecond."}
    assert resolve_sentence(page, 2) == "Second."
