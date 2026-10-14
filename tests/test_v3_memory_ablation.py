from errorflow.v3_memory import build_action_memories
from scripts.evaluate_v3_memory_ablation import (
    outcome_counts,
    population_records,
    shuffle_outcomes,
    summarize_memory_replicates,
)


def baseline(example_id, correct, fold, prediction="SUPPORTED"):
    return {
        "example_id": example_id, "dataset": "AVeriTeC", "split": "train",
        "claim": f"claim {example_id}", "gold_label": "SUPPORTED",
        "prediction": prediction, "correct": correct,
        "diagnosis": {"error_type": "x"}, "label_scores": {},
        "prediction_margin": 0.1, "claim_tokens": 2, "evidence_count": 1,
        "evidence_tokens": 5, "outer_fold": fold,
    }


def candidate(example_id, prediction):
    return {"example_id": example_id, "prediction": prediction, "metadata": {"model_forward_passes": 4}}


def records():
    rows = [
        baseline("a", False, 0), baseline("b", False, 0),
        baseline("c", True, 1), baseline("d", True, 1),
    ]
    candidates = [
        candidate("a", "SUPPORTED"), candidate("b", "REFUTED"),
        candidate("c", "REFUTED"), candidate("d", "SUPPORTED"),
    ]
    return build_action_memories(rows, [("audit", candidates)])


def test_error_and_correct_populations_are_explicitly_separated():
    rows = records()
    error = population_records(rows, "error_only")
    correct = population_records(rows, "correct_only")
    assert len(error) == len(correct) == 2
    assert outcome_counts(error) == {"records": 2, "recovery": 1, "regression": 0, "neutral": 1}
    assert outcome_counts(correct) == {"records": 2, "recovery": 0, "regression": 1, "neutral": 1}


def test_outcome_shuffle_preserves_stratum_marginals_and_is_deterministic():
    rows = records()
    first = shuffle_outcomes(rows, 7)
    again = shuffle_outcomes(rows, 7)
    assert outcome_counts(first) == outcome_counts(rows)
    assert [(r.memory_id, r.recovery_count, r.regression_count) for r in first] == [
        (r.memory_id, r.recovery_count, r.regression_count) for r in again
    ]
    for fold in {0, 1}:
        before = outcome_counts([r for r in rows if r.provenance["outer_fold"] == fold])
        after = outcome_counts([r for r in first if r.provenance["outer_fold"] == fold])
        assert before == after


def test_memory_replicate_summary_uses_b9_overlap_name():
    row = {
        "routed": 1, "route_rate": 0.5, "accuracy": 0.5, "macro_f1": 0.4,
        "helpful": 1, "harmful": 0, "net": 1, "avg_forward_passes": 6.0,
        "delta_accuracy_vs_b1": 0.1, "delta_macro_f1_vs_b1": 0.2,
        "uniform_weight_fallback_count": 0, "uniform_weight_fallback_rate": 0.0,
        "route_overlap_with_b9_natural": {"jaccard": 0.25},
    }
    summary = summarize_memory_replicates([row, row])
    assert summary["route_jaccard_with_b9_natural"]["mean"] == 0.25
    assert "route_jaccard_with_a0_natural" not in summary
