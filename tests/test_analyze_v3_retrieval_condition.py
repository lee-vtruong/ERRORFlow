from scripts.analyze_v3_retrieval_condition import summarize


def prediction(example_id, policy, predicted, gold, routed=False):
    return {
        "example_id": f"scifact-dev:{example_id}", "policy": policy,
        "prediction": predicted, "gold_label": gold, "routed": routed,
    }


def test_stratifies_hit_miss_and_nei_without_changing_predictions():
    claims = [
        {"id": 1, "evidence": {"10": [{"label": "SUPPORT", "sentences": [0]}]}},
        {"id": 2, "evidence": {"20": [{"label": "CONTRADICT", "sentences": [1]}]}},
        {"id": 3, "evidence": {}},
    ]
    retrieved = [
        {"example_id": "scifact-dev:1", "evidence": [{"metadata": {"doc_id": "10", "sentence_id": 0}}]},
        {"example_id": "scifact-dev:2", "evidence": [{"metadata": {"doc_id": "99", "sentence_id": 1}}]},
        {"example_id": "scifact-dev:3", "evidence": []},
    ]
    rows = []
    for example_id, gold in [(1, "SUPPORTED"), (2, "REFUTED"), (3, "NOT ENOUGH INFO")]:
        rows.append(prediction(example_id, "B1", "SUPPORTED", gold))
        rows.append(prediction(example_id, "R6", gold, gold, routed=True))
    result = summarize(claims, retrieved, rows)
    assert result["groups"]["verifiable_hit"]["n"] == 1
    assert result["groups"]["verifiable_miss"]["n"] == 1
    assert result["groups"]["nei_no_gold_rationale"]["n"] == 1
    assert result["groups"]["verifiable_miss"]["systems"]["R6"]["net_vs_b1"] == 1
