from scripts.evaluate_v2_predictions import indexed


def test_indexed_rejects_duplicate_example_ids():
    try:
        indexed([{"example_id": "x"}, {"example_id": "x"}], "rows")
    except ValueError as exc:
        assert "Duplicate example_id" in str(exc)
    else:
        raise AssertionError("duplicate IDs were accepted")
