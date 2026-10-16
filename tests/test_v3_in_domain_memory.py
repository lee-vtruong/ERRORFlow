import json
import sys

from errorflow.io import write_jsonl
from errorflow.v3_memory import ResidualMemoryStore, build_action_memories
from errorflow.v3_statistics import grouped_paired_bootstrap
from scripts.evaluate_v3_in_domain_memory import evaluate, finite_top_count, main


def residual(example_id, fold, correct=False, dataset="Climate-FEVER"):
    return {
        "example_id": example_id,
        "dataset": dataset,
        "split": "all",
        "claim": f"claim {example_id}",
        "gold_label": "SUPPORTED",
        "prediction": "SUPPORTED" if correct else "REFUTED",
        "correct": correct,
        "diagnosis": {"error_type": None if correct else "false_contradiction"},
        "label_scores": {"SUPPORTED": -2.0, "REFUTED": -1.0},
        "prediction_margin": 0.2,
        "claim_tokens": 2,
        "evidence_count": 1,
        "evidence_tokens": 5,
        "model_forward_passes": 4,
        "outer_fold": fold,
    }


def candidate(example_id, prediction="SUPPORTED"):
    return {
        "example_id": example_id,
        "prediction": prediction,
        "metadata": {"model_forward_passes": 4, "llm_requests": 1},
    }


def test_grouped_in_domain_memory_requires_fold_and_excludes_validation_fold():
    rows = [residual("cf:1", 0), residual("cf:2", 1)]
    candidates = [candidate("cf:1"), candidate("cf:2")]
    records = build_action_memories(
        rows,
        [("audit", candidates)],
        dataset="Climate-FEVER",
        source_protocol="grouped_in_domain",
    )
    assert {record.source_dataset for record in records} == {"Climate-FEVER"}
    assert {record.source_split for record in records} == {"train"}
    assert {record.provenance["outer_fold"] for record in records} == {0, 1}
    store = ResidualMemoryStore(records)
    query = {
        "claim": "claim",
        "retrieval_features": {
            "baseline_prediction": "REFUTED",
            "label_scores": {"SUPPORTED": -2.0, "REFUTED": -1.0},
            "evidence_tokens": 5,
        },
    }
    retrieved = store.retrieve(query, action="audit", exclude_folds={0}, compatibility_mode="none")
    assert [record.source_claim_id for record, _ in retrieved] == ["cf:2"]


def test_in_domain_evaluation_uses_protocol_specific_forward_passes():
    rows = {
        "sf:1": {**residual("sf:1", 0, dataset="SciFact"), "model_forward_passes": 3},
        "sf:2": {**residual("sf:2", 1, correct=True, dataset="SciFact"), "model_forward_passes": 3},
    }
    candidates = {
        "sf:1": {**candidate("sf:1"), "metadata": {"model_forward_passes": 3}},
        "sf:2": {**candidate("sf:2", "REFUTED"), "metadata": {"model_forward_passes": 3}},
    }
    metrics, predictions = evaluate(["sf:1", "sf:2"], rows, candidates, {"sf:1"})
    assert predictions == ["SUPPORTED", "SUPPORTED"]
    assert metrics["helpful"] == 1
    assert metrics["harmful"] == 0
    assert metrics["avg_model_forward_passes"] == 4.5


def test_matched_selection_rejects_insufficient_eligible_scores():
    assert finite_top_count(["a", "b"], {"a": 1.0, "b": 0.5}, 1) == {"a"}


def test_grouped_bootstrap_resamples_complete_groups():
    result = grouped_paired_bootstrap(
        ["A", "A", "B"], ["B", "B", "B"], ["A", "A", "B"],
        ["duplicate-pair", "duplicate-pair", "singleton"], samples=20, seed=7,
    )
    assert result["resampling_unit"] == "group"
    assert result["n_groups"] == 2
    assert result["accuracy_delta_ci95"][0] >= 0.0


def test_in_domain_cli_materializes_fold_safe_artifacts(tmp_path, monkeypatch):
    gold_rows, baseline_rows, candidate_rows = [], [], []
    for index in range(10):
        key = f"climate:{index}"
        gold_rows.append({
            "example_id": key,
            "dataset": "Climate-FEVER",
            "split": "all",
            "claim": f"unique{index} climate assertion",
            "label": "SUPPORTED",
            "evidence": [{"evidence_id": "e", "text": "evidence", "source": "test"}],
        })
        baseline_rows.append({
            "example_id": key,
            "prediction": "REFUTED",
            "metadata": {
                "label_scores": {"SUPPORTED": -2.0, "REFUTED": -1.0},
                "model_forward_passes": 4,
            },
        })
        candidate_rows.append({
            "example_id": key,
            "prediction": "SUPPORTED",
            "metadata": {"model_forward_passes": 4, "llm_requests": 1},
        })
    gold_path, baseline_path, candidate_path = (
        tmp_path / "gold.jsonl",
        tmp_path / "baseline.jsonl",
        tmp_path / "candidate.jsonl",
    )
    write_jsonl(gold_path, gold_rows)
    write_jsonl(baseline_path, baseline_rows)
    write_jsonl(candidate_path, candidate_rows)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_v3_in_domain_memory.py",
        "--gold", str(gold_path),
        "--baseline", str(baseline_path),
        "--candidate", str(candidate_path),
        "--output-dir", str(output),
        "--folds", "5",
        "--samples", "20",
        "--random-seeds", "2026",
    ])
    main()
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["results"]["Base"]["accuracy"] == 0.0
    assert report["results"]["Memory-Global-natural"]["accuracy"] == 1.0
    assert all(row["usable_validation_id_overlap"] == 0 for row in report["fold_isolation"])
    assert report["comparisons"]["Memory-Global_vs_Base"]["grouped_paired_bootstrap"]["n_groups"] == 10
    assert report["hypothesis_test"]["resampling_unit"] == "near_duplicate_group"
    assert sum(1 for _ in (output / "memory.jsonl").open(encoding="utf-8")) == 10
