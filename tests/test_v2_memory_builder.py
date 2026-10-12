from errorflow.memory_builder import build_memory_records


def row(claim_id="1"):
    return {"claim_id": claim_id, "claim": "A claim", "gold_label": "REFUTED",
            "baseline_prediction": "SUPPORTED", "final_prediction": "REFUTED",
            "extra_tokens": 10, "llm_calls": 2}


def test_builder_records_only_baseline_errors():
    correct = row("2")
    correct["baseline_prediction"] = "REFUTED"
    records = build_memory_records([row(), correct], dataset="SciFact", source_split="train", action="critic")
    assert len(records) == 1
    assert records[0].recovered is True
    assert records[0].extra_llm_calls == 1


def test_dev_memory_is_non_routable_by_default():
    record = build_memory_records([row()], dataset="SciFact", source_split="dev", action="critic")[0]
    assert record.allow_for_routing is False
    overridden = build_memory_records([row()], dataset="SciFact", source_split="dev", action="critic",
                                      permit_evaluation_routing=True)[0]
    assert overridden.allow_for_routing is True
    assert overridden.metadata["evaluation_memory_override"] is True
