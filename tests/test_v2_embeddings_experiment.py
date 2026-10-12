import pytest

from errorflow.embeddings import HashingEmbedder, cosine
from errorflow.experiment import RunManifest, validate_memory_boundary


def test_hashing_embeddings_are_deterministic_and_normalized():
    embedder = HashingEmbedder(dimensions=32)
    first, second = embedder.encode(["same claim", "same claim"])
    assert first == second
    assert cosine(first, second) == pytest.approx(1.0)


def test_manifest_has_stable_config_identity(tmp_path):
    first = RunManifest("pilot", {"b": 2, "a": 1}, "SciFact", "dev", 3)
    second = RunManifest("pilot", {"a": 1, "b": 2}, "SciFact", "dev", 3)
    assert first.run_id == second.run_id
    assert first.save(tmp_path).exists()


def test_evaluation_memory_boundary_rejects_dev_and_test():
    with pytest.raises(ValueError, match="may not include"):
        validate_memory_boundary("dev", {"train", "dev"})
    validate_memory_boundary("dev", {"train"})
