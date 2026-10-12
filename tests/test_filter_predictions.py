from importlib.machinery import SourceFileLoader

import pytest


MODULE = SourceFileLoader(
    "filter_predictions_by_claims", "scripts/filter_predictions_by_claims.py"
).load_module()


def test_select_predictions_follows_claim_order():
    claims = [{"claim_id": "b"}, {"claim_id": "a"}]
    predictions = [{"claim_id": "a", "prediction": "SUPPORTED"}, {"claim_id": "b", "prediction": "REFUTED"}]
    selected = MODULE.select_predictions(claims, predictions)
    assert [row["claim_id"] for row in selected] == ["b", "a"]


def test_select_predictions_fails_when_prediction_is_missing():
    with pytest.raises(ValueError, match="missing 1 predictions"):
        MODULE.select_predictions([{"claim_id": "a"}], [])
