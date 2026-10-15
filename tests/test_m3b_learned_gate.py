import json
import sys

from errorflow.io import atomic_write_json, write_jsonl
from errorflow.meta_routing.learned_gate import FEATURE_NAMES, feature_vector, fit_logistic
from scripts.evaluate_v3_m3b_learned_gate import main


def test_l2_logistic_gate_learns_simple_separation():
    rows = [(-2, 0, 0, 0, 0, 0), (-1, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0), (2, 0, 0, 0, 0, 0)]
    model = fit_logistic(rows, [0, 0, 1, 1], (0,), l2=0.1)
    assert model.probability((-2,)) < 0.5
    assert model.probability((2,)) > 0.5
    assert len(FEATURE_NAMES) == 6


def _decision(policy, route, score):
    return {"policy": policy, "route": route, "score": score,
            "selected_neighborhood": None, "reason": "test"}


def _make_meta(directory):
    directory.mkdir(parents=True)
    features, predictions = [], []
    policies = (
        "D0_HardLabel", "D1_NLF", "D2_Union", "D3_Intersection",
        "M1_MaxUtility", "M2_Reliability", "M3a_DisagreementGate",
        "M3_ConstantMajority", "M3_ConfidenceControl",
    )
    for index in range(30):
        key = f"train:{index}"
        fold = index % 5
        helpful = index % 2 == 0
        neutral = index in (0, 7, 14, 21, 28)
        gold = "SUPPORTED"
        if neutral:
            b1 = a3 = "REFUTED"
        elif helpful:
            b1, a3 = "REFUTED", "SUPPORTED"
        else:
            b1, a3 = "SUPPORTED", "REFUTED"
        hard_u, nlf_u = (-0.2, 0.8) if helpful else (-0.1, 0.2)
        decisions = {}
        for policy in policies:
            route = policy not in ("D0_HardLabel", "D3_Intersection")
            decisions[policy] = _decision(policy, route, nlf_u if route else hard_u)
        feature = {
            "example_id": key, "outer_fold": fold,
            "baseline_prediction": b1, "baseline_entropy": 0.8 if helpful else 0.2,
            "baseline_top_two_margin": 0.1, "top5_neighbor_overlap": 0.4,
            "hard": {"utility": hard_u, "cross_label_similarity_mass": 0.0,
                     "neighbor_claim_ids": []},
            "nlf": {"utility": nlf_u, "cross_label_similarity_mass": 0.7 if helpful else 0.1,
                    "neighbor_claim_ids": []},
            "decisions": decisions,
        }
        assert len(feature_vector(feature)) == 6
        features.append(feature)
        for policy, prediction in (("B1", b1), ("A3-All", a3)):
            predictions.append({"example_id": key, "mode": "natural", "policy": policy,
                                "routed": policy == "A3-All", "prediction": prediction,
                                "gold_label": gold})
        for seed in range(2026, 2046):
            predictions.append({"example_id": key, "mode": "natural", "policy": "D4_Random",
                                "seed": seed, "routed": bool((index + seed) % 2),
                                "prediction": a3 if (index + seed) % 2 else b1,
                                "gold_label": gold})
    write_jsonl(directory / "per_query_features.jsonl", features)
    write_jsonl(directory / "per_query_predictions.jsonl", predictions)
    atomic_write_json(directory / "method_configs.json", {
        "backbone_id": "synthetic", "protocol": "grouped_internal",
        "source_selected_deployable_beta": 0.0,
    })


def test_m3b_cli_writes_isolated_models_and_predictions(tmp_path, monkeypatch):
    meta = tmp_path / "meta"
    output = tmp_path / "m3b"
    _make_meta(meta)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_v3_m3b_learned_gate.py",
        "--source-meta-dir", str(meta), "--target-meta-dir", str(meta),
        "--output-dir", str(output), "--mode", "source-oof",
        "--bootstrap-samples", "20",
    ])
    main()
    required = {
        "models.json", "summary.json", "statistical_comparisons.json",
        "audit.json", "natural_results.csv", "matched_budget_results.csv",
        "oracle_regret.csv", "per_query_predictions.jsonl",
        "feature_ablation.csv", "experiment_report.md",
    }
    assert required <= {path.name for path in output.iterdir()}
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    models = json.loads((output / "models.json").read_text(encoding="utf-8"))
    assert audit["feature_count"] == 6
    assert not audit["target_labels_used_for_fitting_or_selection"]
    assert models["training_counts"]["neutral"] == 5
    assert set(models["variants"]["M3b_full"]["fold_models"]) == {"0", "1", "2", "3", "4"}
