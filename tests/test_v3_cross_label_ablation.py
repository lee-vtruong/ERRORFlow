import json

from errorflow.io import read_jsonl, write_jsonl
from errorflow.v3_memory import build_action_memories, record_from_dict
from scripts import evaluate_v3_cross_label_ablation as module


def baseline(example_id, prediction, fold):
    correct = prediction == "SUPPORTED"
    return {
        "example_id": example_id,
        "dataset": "AVeriTeC",
        "split": "train",
        "claim": f"claim {example_id}",
        "gold_label": "SUPPORTED",
        "prediction": prediction,
        "correct": correct,
        "diagnosis": {"error_type": "correct" if correct else "missed_support"},
        "label_scores": {"SUPPORTED": 0.7, "REFUTED": 0.3},
        "prediction_margin": 0.4,
        "claim_tokens": 2,
        "evidence_count": 1,
        "evidence_tokens": 5,
        "outer_fold": fold,
        "latency_ms": 1.0,
    }


def candidate(example_id):
    return {
        "example_id": example_id,
        "prediction": "SUPPORTED",
        "latency_ms": 1.0,
        "token_count": 0,
        "metadata": {"model_forward_passes": 4},
    }


def test_contribution_decomposition_and_quota_are_explicit():
    rows = [
        baseline("a", "SUPPORTED", 0),
        baseline("b", "REFUTED", 1),
        baseline("c", "REFUTED", 1),
    ]
    records = build_action_memories(rows, [("audit", [candidate(row["example_id"]) for row in rows])])
    same = [(records[0], 0.8)]
    cross = [(records[1], 0.7), (records[2], 0.6)]
    bundle = module.estimate_bundle(
        same + cross, "audit", "SUPPORTED", min_support=2, cost_weight=0.1
    )
    assert bundle["cross_label_count_at_5"] == 2
    assert abs(bundle["cross_label_similarity_mass"] - 1.3) < 1e-12
    assert abs(
        bundle["score"]
        - bundle["same_label_utility_contribution"]
        - bundle["cross_label_utility_contribution"]
        + bundle["estimated_cost_penalty"]
    ) < 1e-12
    quota = module.quota_neighbors(same, cross, top_k=3, quota=1)
    assert sum(not module.same_label(record, "SUPPORTED") for record, _ in quota) == 1
    suppressed = module.estimate_bundle(
        same, "audit", "SUPPORTED", min_support=2, cost_weight=0.1
    )
    assert suppressed["score"] == float("-inf")
    assert suppressed["support"] == 0
    controlled = module.neutralize_cross_label_outcomes(same + cross, "SUPPORTED")
    controlled_bundle = module.estimate_bundle(
        controlled, "audit", "SUPPORTED", min_support=2, cost_weight=0.1
    )
    assert controlled_bundle["support"] == bundle["support"]
    assert controlled_bundle["cross_label_similarity_mass"] == bundle["cross_label_similarity_mass"]
    assert controlled_bundle["cross_label_utility_contribution"] == 0.0
    assert controlled_bundle["same_label_utility_contribution"] == bundle["same_label_utility_contribution"]


def test_cross_label_cli_writes_mechanistic_and_paired_artifacts(tmp_path, monkeypatch):
    rows = []
    for index in range(8):
        prediction = "SUPPORTED" if (index // 2) % 2 == 0 else "REFUTED"
        rows.append(baseline(f"e{index}", prediction, index % 2))
    candidates = [candidate(row["example_id"]) for row in rows]
    memories = build_action_memories(rows, [("audit", candidates)])
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    memory_path = tmp_path / "memory.jsonl"
    output = tmp_path / "out"
    write_jsonl(baseline_path, rows)
    write_jsonl(candidate_path, candidates)
    write_jsonl(memory_path, (record.to_dict() for record in memories))

    def frozen_grouping(baseline_rows, memory_rows, threshold, folds, seed):
        return (
            baseline_rows,
            [record_from_dict(row) for row in memory_rows],
            {row["example_id"]: row["outer_fold"] for row in baseline_rows},
            {"threshold": threshold, "test_fixture": True},
        )

    monkeypatch.setattr(module, "prepare_grouped_artifacts", frozen_grouping)
    monkeypatch.setattr("sys.argv", [
        "evaluate_v3_cross_label_ablation.py",
        "--baseline-records", str(baseline_path),
        "--candidate", str(candidate_path),
        "--memory", str(memory_path),
        "--output-dir", str(output),
        "--action", "audit",
        "--top-k", "2",
        "--min-support", "1",
        "--folds", "2",
        "--shuffle-seeds", "2026", "2027",
        "--bootstrap-samples", "10",
        "--utility-threshold", "1.0",
        "--seed", "2026",
    ])
    module.main()

    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    mechanism = list(read_jsonl(output / "mechanistic.jsonl"))
    assert report["evaluation"] == "grouped_fold_excluded_cross_label_memory_mechanism"
    assert len(report["primary_results"]) == 20
    assert len(report["matched_significance_nb0_minus_variant"]) == 9
    assert report["nb4_stochastic_aggregate"]["natural"]["seeds"] == 2
    assert len(mechanism) == 8
    required = {
        "cross_label_count@5",
        "cross_label_count_at_5",
        "cross_label_similarity_mass",
        "same_label_utility_contribution",
        "cross_label_utility_contribution",
        "route_changed_vs_hardlabel",
        "helpful_or_harmful_after_route",
    }
    assert all(required <= set(row) for row in mechanism)
    assert (output / "predictions.jsonl").is_file()
    assert (output / "routes.jsonl").is_file()
    assert (output / "RESULTS_FOR_CODEX.txt").is_file()
