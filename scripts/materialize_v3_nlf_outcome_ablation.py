"""Materialize the canonical NA0--NA8 ablation from frozen direct-NLF runs.

The NA taxonomy is a paper-facing relabeling of controls already computed by
``evaluate_v3_memory_ablation.py`` and ``evaluate_v3_memory_controls.py``.
This script performs no model inference and no route selection.  It validates
the source protocol, selects the predeclared primary seed for stochastic
controls, preserves prediction-level rows, and records every source alias.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl, write_jsonl


ALIASES = {
    "NA0_full_r6_nlf": ("base", "B9_full_outcome_memory", None),
    "NA1_no_memory": ("base", "B0_no_memory", None),
    "NA2_random_neighbors": ("base", "B1_random_memory", "primary_seed"),
    "NA3_stratified_outcome_shuffle": (
        "controls", "B14_stratified_outcome_shuffle", "primary_seed"
    ),
    "NA4_global_outcome_prior": ("base", "B6_static_global", None),
    "NA5_class_conditional_prior": ("controls", "B15_class_conditional", None),
    "NA6_recovery_only_utility": ("controls", "B10_recovery_only", None),
    "NA7_error_only_memory": ("base", "B7_error_only", None),
    "NA8_correct_only_memory": ("base", "B8_correct_only", None),
}

SIGNIFICANCE_ALIASES = {
    "NA1_no_memory": ("base", "B0_no_memory"),
    "NA2_random_neighbors": ("base", "B1_random_memory"),
    "NA3_stratified_outcome_shuffle": ("controls", "B14_stratified_outcome_shuffle"),
    "NA4_global_outcome_prior": ("base", "B6_static_global"),
    "NA5_class_conditional_prior": ("controls", "B15_class_conditional"),
    "NA6_recovery_only_utility": ("controls", "B10_recovery_only"),
    "NA7_error_only_memory": ("base", "B7_error_only"),
    "NA8_correct_only_memory": ("base", "B8_correct_only"),
}

BUDGET_ALIASES = {
    "NA0_full_r6_nlf": "B9_full_outcome_memory",
    "NA6_recovery_only_utility": "B10_recovery_only",
    "NA7_error_only_memory": "B7_error_only",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_source(name, report):
    config = report.get("configuration", {})
    if config.get("compatibility_mode") != "none":
        raise ValueError(
            f"{name} is not a direct NLF artifact: "
            f"compatibility_mode={config.get('compatibility_mode')!r}"
        )
    if config.get("matched_route_count") is None:
        raise ValueError(f"{name} does not declare matched_route_count")


def result_key(source_variant, mode, seed_mode, seed):
    suffix = f":seed={seed}" if seed_mode == "primary_seed" else ""
    return f"{source_variant}:{mode}{suffix}"


def prediction_matches(row, source_variant, mode, seed_mode, seed):
    if row.get("variant") != source_variant or row.get("mode") != mode:
        return False
    expected_seed = seed if seed_mode == "primary_seed" else None
    return row.get("seed") == expected_seed


def remap_predictions(paths, seed):
    rows = {"base": list(read_jsonl(paths["base"])),
            "controls": list(read_jsonl(paths["controls"]))}
    output = []
    for alias, (source, source_variant, seed_mode) in ALIASES.items():
        for mode in ("natural", "matched_budget"):
            matched = [
                row for row in rows[source]
                if prediction_matches(row, source_variant, mode, seed_mode, seed)
            ]
            if not matched:
                raise ValueError(f"No prediction rows for {alias}:{mode}")
            for row in matched:
                copied = dict(row)
                copied["source_variant"] = source_variant
                copied["variant"] = alias
                output.append(copied)
    return output


def route_rows(predictions):
    grouped = {}
    for row in predictions:
        key = (row["variant"], row["mode"], row.get("seed"))
        grouped.setdefault(key, []).append(row)
    output = []
    for (variant, mode, seed), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], str(item[0][2]))
    ):
        output.append({
            "variant": variant,
            "mode": mode,
            "seed": seed,
            "selected_ids": sorted(
                str(row["example_id"]) for row in rows if row.get("routed")
            ),
        })
    return output


def remap_budget_predictions(path, seed):
    source_to_alias = {value: key for key, value in BUDGET_ALIASES.items()}
    output = []
    for row in read_jsonl(path):
        alias = source_to_alias.get(row.get("variant"))
        if not alias:
            continue
        copied = dict(row)
        copied["source_variant"] = row["variant"]
        copied["variant"] = alias
        copied["seed"] = seed
        output.append(copied)
    return output


def format_text(report):
    lines = [
        "ERRORFlow V3 direct R6-NLF outcome-memory ablation",
        f"matched_route_count={report['configuration']['matched_route_count']}",
        "",
        "VARIANT\tMODE\tROUTED\tRATE\tHELPFUL\tHARMFUL\tNET\tACC\tMACRO_F1",
    ]
    for alias in ALIASES:
        for mode in ("natural", "matched_budget"):
            row = report["results"][f"{alias}:{mode}"]
            lines.append("\t".join(map(str, (
                alias, mode, row["routed"], f"{row['route_rate']:.6f}",
                row["helpful"], row["harmful"], row["net"],
                f"{row['accuracy']:.6f}", f"{row['macro_f1']:.6f}",
            ))))
    lines.extend(("", "MATCHED SIGNIFICANCE: NA0 MINUS CONTROL"))
    for alias, row in report["matched_significance_na0_minus_control"].items():
        boot = row.get("paired_bootstrap_reference_minus_variant", row.get("paired_bootstrap", {}))
        lines.append(
            f"{alias}\tholm_acc_p={row.get('holm_adjusted_mcnemar_p')}"
            f"\tacc_ci={boot.get('accuracy_delta_ci95')}"
            f"\tf1_ci={boot.get('macro_f1_delta_ci95')}"
        )
    lines.extend(("", "BUDGET RESULTS"))
    for key, row in report["budget_results"].items():
        lines.append(
            f"{key}\trouted={row['routed']}\tacc={row['accuracy']:.6f}"
            f"\tf1={row['macro_f1']:.6f}\tnet={row['net']}"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-summary", required=True, type=Path)
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--control-summary", required=True, type=Path)
    parser.add_argument("--control-predictions", required=True, type=Path)
    parser.add_argument("--control-budget-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    reports = {
        "base": load_json(args.base_summary),
        "controls": load_json(args.control_summary),
    }
    for name, report in reports.items():
        validate_source(name, report)
    matched_counts = {
        report["configuration"]["matched_route_count"] for report in reports.values()
    }
    if len(matched_counts) != 1:
        raise ValueError(f"Source matched-route counts differ: {sorted(matched_counts)}")
    matched_count = matched_counts.pop()

    results = {}
    for alias, (source, source_variant, seed_mode) in ALIASES.items():
        source_results = reports[source].get(
            "all_results", reports[source].get("results", {})
        )
        for mode in ("natural", "matched_budget"):
            key = result_key(source_variant, mode, seed_mode, args.seed)
            if key not in source_results:
                raise KeyError(f"Missing result {source}:{key}")
            results[f"{alias}:{mode}"] = source_results[key]

    significance = {}
    for alias, (source, source_variant) in SIGNIFICANCE_ALIASES.items():
        source_rows = reports[source]["primary_matched_comparisons_b9_minus_variant"]
        significance[alias] = source_rows[source_variant]

    stochastic = {
        "NA2_random_neighbors": reports["base"]["stochastic_aggregate"]["B1_random_memory"],
        "NA3_stratified_outcome_shuffle": reports["controls"]["stochastic_aggregate"][
            "B14_stratified_outcome_shuffle"
        ],
    }
    budget_results = {}
    source_budget = reports["controls"]["budget_results"]
    for alias, source_variant in BUDGET_ALIASES.items():
        prefix = f"{source_variant}:budget="
        for key, row in source_budget.items():
            if key.startswith(prefix):
                budget = key.split("budget=", 1)[1]
                budget_results[f"{alias}:budget={budget}"] = row

    report = {
        "evaluation": "canonical_direct_r6_nlf_outcome_memory_ablation",
        "configuration": {
            "compatibility_mode": "none",
            "matched_route_count": matched_count,
            "matched_route_rate": matched_count / reports["base"]["baseline"]["n"],
            "seed": args.seed,
            "stochastic_seeds": reports["base"]["configuration"]["random_seeds"],
            "budget_grid": reports["controls"]["configuration"]["budgets"],
            "materialization_only": True,
        },
        "aliases": {
            alias: {"source": source, "source_variant": variant,
                    "seed_mode": seed_mode}
            for alias, (source, variant, seed_mode) in ALIASES.items()
        },
        "baseline": reports["base"]["baseline"],
        "results": results,
        "stochastic_aggregate": stochastic,
        "matched_significance_na0_minus_control": significance,
        "budget_results": budget_results,
        "integrity_notes": [
            "No model inference or route selection is performed by this script.",
            "NA labels are aliases of frozen direct-NLF B controls.",
            "NA2 and NA3 paired tests use the predeclared seed; aggregates use 20 seeds.",
            "NA7 budget analysis reuses exactly matched route counts from the frozen sweep.",
        ],
    }

    predictions = remap_predictions(
        {"base": args.base_predictions, "controls": args.control_predictions}, args.seed
    )
    budget_predictions = remap_budget_predictions(
        args.control_budget_predictions, args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", report)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_jsonl(args.output_dir / "routes.jsonl", route_rows(predictions))
    write_jsonl(args.output_dir / "budget_predictions.jsonl", budget_predictions)
    (args.output_dir / "RESULTS_FOR_CODEX.txt").write_text(
        format_text(report), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "variants": len(ALIASES),
        "prediction_rows": len(predictions),
        "budget_prediction_rows": len(budget_predictions),
        "matched_route_count": matched_count,
    }, indent=2))


if __name__ == "__main__":
    main()
