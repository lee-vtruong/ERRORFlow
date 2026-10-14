from dataclasses import replace

from scripts.evaluate_v3_memory_controls import (
    estimate_neighbors,
    outcome_bucket,
    stable_sample,
    stratified_outcome_sample,
    summarize_seed_rows,
)
from errorflow.v3_memory import build_action_memories
from errorflow.io import write_jsonl


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
    return {
        "example_id": example_id,
        "prediction": prediction,
        "metadata": {"model_forward_passes": 4},
    }


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


def test_recovery_and_regression_only_scores_remove_the_intended_term():
    memory = records()
    helpful = next(row for row in memory if row.recovery_count)
    harmful = next(row for row in memory if row.regression_count)
    neighbors = [(helpful, 1.0), (harmful, 1.0)]
    full, support, _, _ = estimate_neighbors(neighbors, "audit", 0.1, "full")
    recovery, _, _, _ = estimate_neighbors(neighbors, "audit", 0.1, "recovery_only")
    safe, _, _, _ = estimate_neighbors(neighbors, "audit", 0.1, "regression_only")
    assert support == 2
    assert recovery > full
    assert safe > full


def test_uniform_outcome_weighting_ignores_similarity_magnitude():
    memory = records()
    helpful = next(row for row in memory if row.recovery_count)
    harmful = next(row for row in memory if row.regression_count)
    left = [(helpful, 0.9), (harmful, 0.1)]
    right = [(helpful, 0.1), (harmful, 0.9)]
    first = estimate_neighbors(left, "audit", 0.1, uniform_outcome_weights=True)[0]
    second = estimate_neighbors(right, "audit", 0.1, uniform_outcome_weights=True)[0]
    assert first == second


def test_size_matched_samples_are_exact_reproducible_and_outcome_mixed():
    base = records()
    expanded = []
    for index in range(20):
        source = base[index % len(base)]
        expanded.append(replace(
            source,
            memory_id=f"m{index}",
            source_claim_id=f"c{index}",
        ))
    first = stable_sample(expanded, 9, 2026)
    again = stable_sample(list(reversed(expanded)), 9, 2026)
    assert [row.memory_id for row in first] == [row.memory_id for row in again]
    stratified = stratified_outcome_sample(expanded, 9, 2026)
    assert len(stratified) == 9
    assert set(map(outcome_bucket, stratified)) == {"recovery", "regression", "neutral"}


def test_seed_summary_reports_standard_deviation():
    template = {
        "routed": 1, "route_rate": 0.1, "accuracy": 0.6,
        "macro_f1": 0.5, "helpful": 2, "harmful": 1, "net": 1,
    }
    other = dict(template, accuracy=0.8, net=3)
    summary = summarize_seed_rows([template, other])
    assert summary["accuracy"]["mean"] == 0.7
    assert summary["accuracy"]["std"] > 0
    assert summary["net"]["min"] == 1
    assert summary["net"]["max"] == 3


def test_extended_controls_cli_smoke_writes_all_artifacts(tmp_path, monkeypatch):
    from scripts.evaluate_v3_memory_controls import main

    baseline_rows = [
        baseline("a", False, 0), baseline("b", False, 0, "REFUTED"),
        baseline("c", True, 1), baseline("d", True, 1, "REFUTED"),
    ]
    candidate_rows = [
        candidate("a", "SUPPORTED"), candidate("b", "SUPPORTED"),
        candidate("c", "REFUTED"), candidate("d", "SUPPORTED"),
    ]
    memory = build_action_memories(baseline_rows, [("audit", candidate_rows)])
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    memory_path = tmp_path / "memory.jsonl"
    output = tmp_path / "out"
    write_jsonl(baseline_path, baseline_rows)
    write_jsonl(candidate_path, candidate_rows)
    write_jsonl(memory_path, (row.to_dict() for row in memory))
    monkeypatch.setattr("sys.argv", [
        "evaluate_v3_memory_controls.py",
        "--baseline-records", str(baseline_path),
        "--candidate", str(candidate_path),
        "--memory", str(memory_path),
        "--output-dir", str(output),
        "--action", "audit",
        "--top-k", "2",
        "--min-support", "1",
        "--folds", "2",
        "--random-seeds", "2026",
        "--size-match-count", "2",
        "--budgets", "0.0",
        "--bootstrap-samples", "10",
    ])
    main()
    assert (output / "summary.json").is_file()
    assert (output / "routes.jsonl").is_file()
    assert (output / "predictions.jsonl").is_file()
    assert (output / "budget_predictions.jsonl").is_file()
