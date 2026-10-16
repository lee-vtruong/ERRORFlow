from scripts.hash_model_snapshot import aggregate_digest, is_model_artifact


def test_aggregate_digest_is_order_invariant():
    rows = [
        {"path": "b", "bytes": 2, "sha256": "bb"},
        {"path": "a", "bytes": 1, "sha256": "aa"},
    ]
    assert aggregate_digest(rows) == aggregate_digest(list(reversed(rows)))


def test_mutable_huggingface_cache_is_not_a_model_artifact():
    assert not is_model_artifact({"path": ".cache/huggingface/download/config.json.lock"})
    assert is_model_artifact({"path": "model-00001-of-00003.safetensors"})
