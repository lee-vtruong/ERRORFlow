import json

from errorflow.io import read_jsonl, write_jsonl
from scripts.materialize_v3_nlf_outcome_ablation import ALIASES, main


def metric(routed=1):
    return {
        "n": 2,
        "routed": routed,
        "route_rate": routed / 2,
        "helpful": routed,
        "harmful": 0,
        "net": routed,
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "avg_model_forward_passes": 4.0 + 4.0 * routed / 2,
    }


def significance():
    return {
        "orientation": "B9_minus_variant",
        "mcnemar_exact": {"exact_two_sided_p": 1.0},
        "paired_bootstrap": {
            "accuracy_delta_ci95": [0.0, 0.0],
            "macro_f1_delta_ci95": [0.0, 0.0],
        },
        "holm_adjusted_mcnemar_p": 1.0,
    }


def prediction_rows(variants):
    rows = []
    for variant, seeded in variants:
        for mode in ("natural", "matched_budget"):
            for index in range(2):
                rows.append({
                    "example_id": f"e{index}",
                    "variant": variant,
                    "mode": mode,
                    "seed": 2026 if seeded else None,
                    "routed": index == 0,
                    "gold_label": "SUPPORTED",
                    "prediction": "SUPPORTED",
                })
    return rows


def test_materializer_exports_complete_direct_nlf_bundle(tmp_path, monkeypatch):
    base_variants = [
        ("B9_full_outcome_memory", False),
        ("B0_no_memory", False),
        ("B1_random_memory", True),
        ("B6_static_global", False),
        ("B7_error_only", False),
        ("B8_correct_only", False),
    ]
    control_variants = [
        ("B14_stratified_outcome_shuffle", True),
        ("B15_class_conditional", False),
        ("B10_recovery_only", False),
    ]
    base_results = {}
    for variant, seeded in base_variants:
        for mode in ("natural", "matched_budget"):
            suffix = ":seed=2026" if seeded else ""
            base_results[f"{variant}:{mode}{suffix}"] = metric()
    control_results = {}
    for variant, seeded in control_variants:
        for mode in ("natural", "matched_budget"):
            suffix = ":seed=2026" if seeded else ""
            control_results[f"{variant}:{mode}{suffix}"] = metric()

    base_report = {
        "configuration": {
            "compatibility_mode": "none",
            "matched_route_count": 1,
            "random_seeds": [2026],
        },
        "baseline": {"n": 2, "accuracy": 0.5, "macro_f1": 0.5},
        "all_results": base_results,
        "stochastic_aggregate": {"B1_random_memory": {"seeds": 1}},
        "primary_matched_comparisons_b9_minus_variant": {
            variant: significance()
            for variant in (
                "B0_no_memory", "B1_random_memory", "B6_static_global",
                "B7_error_only", "B8_correct_only",
            )
        },
    }
    control_report = {
        "configuration": {
            "compatibility_mode": "none",
            "matched_route_count": 1,
            "budgets": [0.1],
        },
        "results": control_results,
        "stochastic_aggregate": {
            "B14_stratified_outcome_shuffle": {"seeds": 1}
        },
        "primary_matched_comparisons_b9_minus_variant": {
            variant: significance()
            for variant in (
                "B14_stratified_outcome_shuffle", "B15_class_conditional",
                "B10_recovery_only",
            )
        },
        "budget_results": {
            f"{variant}:budget=0.1000": metric()
            for variant in (
                "B9_full_outcome_memory", "B10_recovery_only", "B7_error_only"
            )
        },
    }

    base_summary = tmp_path / "base.json"
    control_summary = tmp_path / "controls.json"
    base_predictions = tmp_path / "base.jsonl"
    control_predictions = tmp_path / "controls.jsonl"
    budget_predictions = tmp_path / "budget.jsonl"
    output = tmp_path / "out"
    base_summary.write_text(json.dumps(base_report), encoding="utf-8")
    control_summary.write_text(json.dumps(control_report), encoding="utf-8")
    write_jsonl(base_predictions, prediction_rows(base_variants))
    write_jsonl(control_predictions, prediction_rows(control_variants))
    write_jsonl(budget_predictions, (
        {
            "example_id": f"e{index}",
            "variant": variant,
            "budget": 0.1,
            "routed": index == 0,
            "prediction": "SUPPORTED",
        }
        for variant in (
            "B9_full_outcome_memory", "B10_recovery_only", "B7_error_only"
        )
        for index in range(2)
    ))

    monkeypatch.setattr("sys.argv", [
        "materialize_v3_nlf_outcome_ablation.py",
        "--base-summary", str(base_summary),
        "--base-predictions", str(base_predictions),
        "--control-summary", str(control_summary),
        "--control-predictions", str(control_predictions),
        "--control-budget-predictions", str(budget_predictions),
        "--output-dir", str(output),
        "--seed", "2026",
    ])
    main()

    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["configuration"]["compatibility_mode"] == "none"
    assert report["configuration"]["materialization_only"] is True
    assert len(report["aliases"]) == 9
    assert len(report["results"]) == 18
    assert len(report["matched_significance_na0_minus_control"]) == 8
    assert len(report["budget_results"]) == 3
    assert len(list(read_jsonl(output / "predictions.jsonl"))) == 36
    assert len(list(read_jsonl(output / "routes.jsonl"))) == 18
    assert len(list(read_jsonl(output / "budget_predictions.jsonl"))) == 6
    assert set(report["aliases"]) == set(ALIASES)


def test_materializer_rejects_hard_label_sources(tmp_path, monkeypatch):
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({
        "configuration": {
            "compatibility_mode": "same_label",
            "matched_route_count": 1,
        }
    }), encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "materialize_v3_nlf_outcome_ablation.py",
        "--base-summary", str(invalid),
        "--base-predictions", str(tmp_path / "missing.jsonl"),
        "--control-summary", str(invalid),
        "--control-predictions", str(tmp_path / "missing2.jsonl"),
        "--control-budget-predictions", str(tmp_path / "missing3.jsonl"),
        "--output-dir", str(tmp_path / "out"),
    ])
    try:
        main()
    except ValueError as exc:
        assert "not a direct NLF artifact" in str(exc)
    else:
        raise AssertionError("hard-label artifacts must be rejected")
