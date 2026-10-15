import json
import sys

from errorflow.io import write_jsonl
from errorflow.v3_memory import ResidualMemoryRecord, ResidualMemoryStore
from scripts.evaluate_v3_meta_routing import main


def test_meta_routing_cli_materializes_required_artifacts(tmp_path, monkeypatch):
    labels = ["SUPPORTED", "REFUTED", "SUPPORTED", "REFUTED"]
    gold, baseline, candidate, memories = [], [], [], []
    for index, label in enumerate(labels):
        key = f"train:{index}"
        prediction = "REFUTED" if label == "SUPPORTED" else "SUPPORTED"
        after = label if index != 3 else prediction
        claim = f"unique claim token{index}"
        gold.append({"example_id": key, "claim": claim, "label": label, "dataset": "AVeriTeC"})
        baseline.append({
            "example_id": key, "claim": claim, "prediction": prediction,
            "gold_label": label, "split": "train", "correct": False,
            "label_scores": {"SUPPORTED": -0.2, "REFUTED": -0.3},
            "evidence_tokens": 10, "outer_fold": index % 2,
        })
        candidate.append({"example_id": key, "prediction": after})
        memories.append(ResidualMemoryRecord(
            memory_id=f"m{index}", version=1, source_dataset="AVeriTeC",
            source_split="train", source_claim_id=key, source_claim=claim,
            error_pattern="test", applicability_conditions=[],
            correction_strategy="conflicting_rescue_v1",
            retrieval_features={
                "baseline_prediction": prediction,
                "label_scores": {"SUPPORTED": -0.2, "REFUTED": -0.3},
                "evidence_tokens": 10,
            },
            training_support=1, recovery_count=int(after == label), regression_count=0,
            estimated_net_utility=float(after == label), estimated_extra_cost=1.0,
            validation_status="test", provenance={"outer_fold": index % 2},
        ))
    gold_path, baseline_path = tmp_path / "gold.jsonl", tmp_path / "b1.jsonl"
    candidate_path, memory_path = tmp_path / "a3.jsonl", tmp_path / "memory.jsonl"
    output = tmp_path / "out"
    write_jsonl(gold_path, gold)
    write_jsonl(baseline_path, baseline)
    write_jsonl(candidate_path, candidate)
    ResidualMemoryStore(memories).save(memory_path)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_v3_meta_routing.py", "--gold", str(gold_path),
        "--baseline", str(baseline_path), "--candidate", str(candidate_path),
        "--memory", str(memory_path), "--output-dir", str(output),
        "--backbone-id", "test-backbone", "--dataset-protocol", "test-grouped",
        "--protocol", "grouped_internal", "--top-k", "1", "--min-support", "1",
        "--folds", "2", "--group-threshold", "1.0", "--bootstrap-samples", "10",
        "--random-seeds", "1", "2", "--beta-grid", "0", "0.5",
    ])
    main()
    required = {
        "audit_report.md", "oracle_headroom.json", "disagreement_analysis.csv",
        "method_configs.json", "natural_results.csv", "matched_budget_results.csv",
        "statistical_comparisons.json", "per_class_results.csv",
        "per_query_predictions.jsonl", "per_query_features.jsonl",
        "quality_budget_curves.csv", "runtime_comparison.csv", "experiment_report.md",
    }
    assert required <= {path.name for path in output.iterdir()}
    feature_rows = [json.loads(line) for line in (output / "per_query_features.jsonl").read_text().splitlines()]
    assert len(feature_rows) == 4
    assert all("gold" not in json.dumps(row).lower() for row in feature_rows)
