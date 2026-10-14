from dataclasses import replace

from errorflow.v3_memory import ResidualMemoryRecord, ResidualMemoryStore
from scripts.evaluate_v3_component_ablation import route_overlap, summarize_replicates, top_count


def record(memory_id, claim):
    return ResidualMemoryRecord(
        memory_id=memory_id, version=1, source_dataset="x", source_split="train",
        source_claim_id=memory_id, source_claim=claim, error_pattern="x",
        applicability_conditions=[], correction_strategy="audit",
        retrieval_features={
            "baseline_prediction": "SUPPORTED",
            "label_scores": {"SUPPORTED": 1.0, "REFUTED": 0.0},
            "evidence_tokens": 10,
        },
        training_support=1, recovery_count=1, regression_count=0,
        estimated_net_utility=1.0, estimated_extra_cost=1.0,
        validation_status="test", provenance={"outer_fold": 1},
    )


def test_similarity_weights_are_normalized_and_can_remove_claim_signal():
    store = ResidualMemoryStore([record("a", "same words"), record("b", "other")])
    query = {
        "example_id": "q", "claim": "same words",
        "retrieval_features": {
            "baseline_prediction": "SUPPORTED",
            "label_scores": {"SUPPORTED": 1.0, "REFUTED": 0.0},
            "evidence_tokens": 10,
        },
    }
    full = store.retrieve(query, top_k=2, similarity_weights=(0.35, 0.50, 0.15))
    no_claim = store.retrieve(query, top_k=2, similarity_weights=(0.0, 5.0, 1.5))
    assert full[0][0].memory_id == "a"
    assert no_claim[0][1] == no_claim[1][1]


def test_random_neighbor_selection_is_seeded_and_query_specific():
    store = ResidualMemoryStore([record(str(index), f"claim {index}") for index in range(10)])
    query = {
        "example_id": "q", "claim": "claim",
        "retrieval_features": {
            "baseline_prediction": "SUPPORTED", "label_scores": {}, "evidence_tokens": 10,
        },
    }
    first = store.retrieve(query, top_k=4, selection_mode="random", random_seed=7)
    again = store.retrieve(query, top_k=4, selection_mode="random", random_seed=7)
    other = store.retrieve(query, top_k=4, selection_mode="random", random_seed=8)
    assert [row.memory_id for row, _ in first] == [row.memory_id for row, _ in again]
    assert [row.memory_id for row, _ in first] != [row.memory_id for row, _ in other]


def test_no_label_filter_can_return_cross_label_neighbors():
    supported = record("supported", "same words")
    features = dict(supported.retrieval_features)
    features["baseline_prediction"] = "REFUTED"
    refuted = replace(
        record("refuted", "same words"), retrieval_features=features,
    )
    store = ResidualMemoryStore([supported, refuted])
    query = {
        "example_id": "q", "claim": "same words",
        "retrieval_features": {
            "baseline_prediction": "SUPPORTED",
            "label_scores": {"SUPPORTED": 1.0, "REFUTED": 0.0},
            "evidence_tokens": 10,
        },
    }
    hard = store.retrieve(query, top_k=2, compatibility_mode="same_label")
    nlf = store.retrieve(query, top_k=2, compatibility_mode="none")
    assert [item.retrieval_features["baseline_prediction"] for item, _ in hard] == ["SUPPORTED"]
    assert {item.retrieval_features["baseline_prediction"] for item, _ in nlf} == {"SUPPORTED", "REFUTED"}


def test_matched_count_and_overlap_are_deterministic():
    scores = {"a": 0.2, "b": 0.2, "c": -0.1}
    assert top_count(sorted(scores), scores, 2) == {"a", "b"}
    assert route_overlap({"a", "b"}, {"b", "c"}) == {
        "intersection": 1, "union": 3, "jaccard": 1 / 3,
    }


def test_random_replicate_summary_includes_cost_and_route_overlap():
    base = {
        "routed": 1, "route_rate": 0.5, "accuracy": 0.5, "macro_f1": 0.4,
        "helpful": 1, "harmful": 0, "net": 1, "avg_forward_passes": 6.0,
        "delta_accuracy_vs_b1": 0.1, "delta_macro_f1_vs_b1": 0.2,
        "uniform_weight_fallback_count": 2, "uniform_weight_fallback_rate": 0.1,
        "route_overlap_with_a0_natural": {"jaccard": 0.25},
    }
    summary = summarize_replicates([base, base])
    assert summary["avg_forward_passes"]["mean"] == 6.0
    assert summary["route_jaccard_with_a0_natural"]["mean"] == 0.25
