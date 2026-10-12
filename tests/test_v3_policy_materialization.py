from errorflow.v3_memory import ResidualMemoryStore, build_action_memories
from scripts.materialize_v3_utility_policy import materialize, summarize


def baseline(index, fold):
    return {
        "example_id": f"train:{index}", "split": "train", "claim": "same claim",
        "gold_label": "REFUTED", "prediction": "SUPPORTED", "correct": False,
        "diagnosis": {"error_type": "missed_contradiction"},
        "label_scores": {"SUPPORTED": -1.0, "REFUTED": -1.5},
        "evidence_tokens": 10, "evidence_count": 1, "claim_tokens": 2,
        "prediction_margin": 0.5, "outer_fold": fold,
    }


def candidate(index):
    return {"example_id": f"train:{index}", "prediction": "REFUTED", "metadata": {"model_forward_passes": 4}}


def test_materialized_utility_policy_excludes_fold_and_routes_positive_neighbors():
    baselines = [baseline(0, 0), baseline(1, 1), baseline(2, 2)]
    candidates = [candidate(0), candidate(1), candidate(2)]
    memory = ResidualMemoryStore(build_action_memories(baselines, [("audit", candidates)]))
    rows = materialize(baselines, candidates, memory, "audit", top_k=2, min_support=2, progress_every=0)
    assert all(row["routed"] for row in rows)
    assert all(row["metadata"]["memory_fold_excluded"] for row in rows)
    assert summarize(rows)["net"] == 3
