from errorflow.verifier import Qwen3TransformersVerifier, VerifierConfig


def test_qwen_adapter_is_lazy():
    verifier = Qwen3TransformersVerifier(VerifierConfig("local-model"))
    assert verifier._model is None
    assert verifier._tokenizer is None
