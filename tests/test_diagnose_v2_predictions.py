from scripts.diagnose_v2_predictions import diagnose


def test_diagnose_detects_generation_limit_and_errors():
    gold = [{"example_id": "dev:0", "label": "CONFLICTING"}]
    predictions = [{
        "example_id": "dev:0", "prediction": "REFUTED", "token_count": 32,
        "metadata": {"raw_output": "reasoning cut off", "confidence_type": "first_generated_token_max_probability"},
    }]
    report = diagnose(gold, predictions, 32)
    assert report["at_generation_limit_rate"] == 1.0
    assert report["prediction_distribution"] == {"REFUTED": 1}
    assert report["errors"][0]["gold_label"] == "CONFLICTING"
