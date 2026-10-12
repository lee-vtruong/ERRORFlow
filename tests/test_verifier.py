import pytest

from errorflow.verifier import Verifier, VerifierConfig, build_prompt, parse_label


def test_prompt_and_label_parser():
    prompt = build_prompt("A claim", ["Evidence one"])
    assert "Label:" in prompt
    assert parse_label("The answer is REFUTED.") == "REFUTED"
    assert "directly" in build_prompt("A claim", ["Evidence"], "check directly")


def test_verifier_record_keeps_provenance():
    verifier = Verifier(VerifierConfig("Qwen/Qwen3-4B-Instruct-2507", seed=42))
    record = verifier.record("c1", "claim", ["e"], "SUPPORTED", split="train")
    assert record.model.startswith("Qwen/")
    assert record.seed == 42
    assert record.prediction == "SUPPORTED"


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError):
        parse_label("uncertain")


def test_averitec_four_label_prompt_and_parser():
    labels = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO", "CONFLICTING")
    assert parse_label("Label: CONFLICTING", labels) == "CONFLICTING"
    assert "CONFLICTING" in build_prompt("claim", ["evidence"], label_space=labels)


def test_label_scoring_api_is_available_without_eager_loading():
    verifier = VerifierConfig("local", label_space=("SUPPORTED", "REFUTED", "CONFLICTING"))
    assert verifier.label_space[-1] == "CONFLICTING"
