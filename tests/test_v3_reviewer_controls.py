import json
import sys

from errorflow.io import write_jsonl
from scripts.evaluate_v3_quality_budget_controls import main as curve_main
from scripts.evaluate_v3_score_ensemble import main as ensemble_main


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
         "prediction": gold[index]["label"]}
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
    assert set(report["weight_selection"]) == {"0", "1", "deployable_source"}


def test_quality_budget_controls(tmp_path, monkeypatch):
    gold, b1, a3, features = rows()
    for name, payload in (("gold", gold), ("b1", b1), ("a3", a3)):
        write_jsonl(tmp_path / f"{name}.jsonl", payload)
    meta = tmp_path / "meta"
    meta.mkdir()
    write_jsonl(meta / "per_query_features.jsonl", features)
    out = tmp_path / "curves"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_v3_quality_budget_controls.py", "--gold", str(tmp_path / "gold.jsonl"),
        "--baseline", str(tmp_path / "b1.jsonl"), "--candidate", str(tmp_path / "a3.jsonl"),
        "--meta-dir", str(meta), "--output-dir", str(out),
        "--budgets", "0", "0.5", "1", "--random-seeds", "1", "2",
    ])
    curve_main()
    report = json.loads((out / "summary.json").read_text())
    assert report["oracle"]["helpful_actions_available"] == 4
    assert report["natural_nlf"]["fraction_of_oracle_helpful_gain"] == 1.0
    assert (out / "quality_budget_curves.csv").exists()
