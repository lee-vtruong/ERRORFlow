import json
import sys

from errorflow.io import write_jsonl
from scripts.evaluate_v3_quality_budget_controls import main as curve_main
from scripts.evaluate_v3_score_ensemble import main as ensemble_main
from scripts.evaluate_v3_supervised_controls import main as supervised_main
from scripts.select_v3_compatibility_rule import main as compatibility_main


def rows():
    gold, b1, a3, features = [], [], [], []
    labels = ["SUPPORTED", "REFUTED", "SUPPORTED", "REFUTED"]
    for index, label in enumerate(labels):
        key = f"x:{index}"
        other = "REFUTED" if label == "SUPPORTED" else "SUPPORTED"
        gold.append({"example_id": key, "label": label})
        b1.append({
            "example_id": key, "prediction": other,
            "metadata": {"label_scores": {label: -1.0, other: -0.1}},
        })
        a3.append({
            "example_id": key, "prediction": label,
            "metadata": {"label_scores": {label: -0.1, other: -1.0}},
        })
        features.append({
            "example_id": key, "outer_fold": index % 2,
            "hard": {"utility": 0.1 + index, "eligible": True},
            "nlf": {"utility": 0.2 + index, "eligible": True},
        })
    return gold, b1, a3, features


def test_score_ensemble_grouped(tmp_path, monkeypatch):
    gold, b1, a3, features = rows()
    for name, payload in (("gold", gold), ("b1", b1), ("a3", a3)):
        write_jsonl(tmp_path / f"{name}.jsonl", payload)
    meta = tmp_path / "meta"
    meta.mkdir()
    write_jsonl(meta / "per_query_features.jsonl", features)
    write_jsonl(meta / "per_query_predictions.jsonl", [
        {"example_id": row["example_id"], "mode": "natural", "policy": "D1_NLF",
         "prediction": gold[index]["label"], "routed": True}
        for index, row in enumerate(features)
    ])
    out = tmp_path / "ensemble"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_v3_score_ensemble.py", "--gold", str(tmp_path / "gold.jsonl"),
        "--baseline", str(tmp_path / "b1.jsonl"), "--candidate", str(tmp_path / "a3.jsonl"),
        "--output-dir", str(out), "--protocol", "grouped_internal", "--meta-dir", str(meta),
        "--alpha-grid", "0", "0.5", "1", "--bootstrap-samples", "20",
    ])
    ensemble_main()
    report = json.loads((out / "summary.json").read_text())
    assert report["results"]["Score-Ensemble"]["accuracy"] == 0.5
    assert report["results"]["Global-neighbor-memory"]["accuracy"] == 1.0
    assert report["results"]["Routed-Score-Fusion"]["accuracy"] == 0.5
    assert report["results"]["Routed-Score-Fusion"]["routed"] == 4
    assert set(report["weight_selection"]) == {"0", "1", "deployable_source"}


def test_quality_budget_controls(tmp_path, monkeypatch):
    gold, b1, a3, features = rows()
    for name, payload in (("gold", gold), ("b1", b1), ("a3", a3)):
        write_jsonl(tmp_path / f"{name}.jsonl", payload)
    meta = tmp_path / "meta"
    meta.mkdir()
    write_jsonl(meta / "per_query_features.jsonl", features)
    supervised = tmp_path / "supervised.jsonl"
    write_jsonl(supervised, [
        {"example_id": row["example_id"], "method": "Logistic", "score": float(index)}
        for index, row in enumerate(features)
    ])
    ensemble = tmp_path / "ensemble.jsonl"
    write_jsonl(ensemble, [
        {"example_id": row["example_id"], "prediction": gold[index]["label"]}
        for index, row in enumerate(features)
    ])
    out = tmp_path / "curves"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_v3_quality_budget_controls.py", "--gold", str(tmp_path / "gold.jsonl"),
        "--baseline", str(tmp_path / "b1.jsonl"), "--candidate", str(tmp_path / "a3.jsonl"),
        "--meta-dir", str(meta), "--output-dir", str(out),
        "--budgets", "0", "0.5", "1", "--random-seeds", "1", "2",
        "--supervised-predictions", str(supervised),
        "--supervised-methods", "Logistic",
        "--ensemble-predictions", str(ensemble),
    ])
    curve_main()
    report = json.loads((out / "summary.json").read_text())
    assert report["oracle"]["helpful_actions_available"] == 4
    assert report["natural_nlf"]["fraction_of_oracle_helpful_gain"] == 1.0
    assert report["supervised_methods"] == ["Logistic"]
    assert report["score_ensemble_included"] is True
    assert (out / "quality_budget_curves.csv").exists()
    csv_lines = (out / "quality_budget_curves.csv").read_text().splitlines()
    assert any("Score-Ensemble" in row for row in csv_lines)


def test_supervised_controls_grouped(tmp_path, monkeypatch):
    gold, b1, a3, features = rows()
    for index, row in enumerate(gold):
        row["claim"] = f"claim {index}"
        row["evidence"] = [{"text": f"evidence {index}"}]
    for name, payload in (("gold", gold), ("b1", b1), ("a3", a3)):
        write_jsonl(tmp_path / f"{name}.jsonl", payload)
    meta = tmp_path / "meta"
    meta.mkdir()
    write_jsonl(meta / "per_query_features.jsonl", features)
    write_jsonl(meta / "per_query_predictions.jsonl", [
        {"example_id": row["example_id"], "mode": "natural", "policy": "D1_NLF",
         "prediction": gold[index]["label"], "routed": index < 2}
        for index, row in enumerate(features)
    ])
    out = tmp_path / "supervised"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_v3_supervised_controls.py", "--gold", str(tmp_path / "gold.jsonl"),
        "--baseline", str(tmp_path / "b1.jsonl"), "--candidate", str(tmp_path / "a3.jsonl"),
        "--meta-dir", str(meta), "--protocol", "grouped_internal",
        "--route-count", "2", "--bootstrap-samples", "20", "--output-dir", str(out),
    ])
    supervised_main()
    report = json.loads((out / "summary.json").read_text())
    assert report["target_labels_used_for_fitting_or_selection"] is False
    assert set(report["results"]) == {
        "Memory-Global", "Logistic", "Logistic-BalancedOutcome",
        "Logistic-PerLabelRank", "Logistic-Interactions", "Boosted-Stumps"
    }
    assert all(result["routed"] == 2 for result in report["results"].values())
    assert all("per_class" in result for result in report["results"].values())
    assert all(
        "holm_accuracy_p" in comparison
        for comparison in report["comparisons"].values()
    )
    assert all(
        "bonferroni_familywise_interval" in comparison
        for comparison in report["comparisons"].values()
    )


def test_group_safe_compatibility_selection(tmp_path, monkeypatch):
    gold, _, _, features = rows()
    write_jsonl(tmp_path / "gold.jsonl", gold)
    meta = tmp_path / "meta"
    meta.mkdir()
    write_jsonl(meta / "per_query_features.jsonl", features)
    predictions = []
    for row, target in zip(features, gold):
        key = row["example_id"]
        wrong = "REFUTED" if target["label"] == "SUPPORTED" else "SUPPORTED"
        predictions.extend([
            {"example_id": key, "mode": "natural", "policy": "D0_HardLabel",
             "prediction": wrong, "routed": False},
            {"example_id": key, "mode": "natural", "policy": "D1_NLF",
             "prediction": target["label"], "routed": True},
        ])
    write_jsonl(meta / "per_query_predictions.jsonl", predictions)
    out = tmp_path / "compatibility"
    monkeypatch.setattr(sys, "argv", [
        "select_v3_compatibility_rule.py", "--gold", str(tmp_path / "gold.jsonl"),
        "--meta-dir", str(meta), "--output-dir", str(out),
        "--bootstrap-samples", "20",
    ])
    compatibility_main()
    report = json.loads((out / "summary.json").read_text())
    assert report["all_source_deployment_choice"]["selected_rule"] == "Memory-Global"
    assert report["cross_fitted_selection"]["accuracy"] == 1.0
    assert all(not row["held_out_labels_used_for_selection"] for row in report["folds"])
