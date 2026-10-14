"""Evaluate N-D utility/decision ablations for the final R6-NLF router.

The verifier is never called.  Every result replays cached B1 and A3
predictions, with grouped outer-fold exclusion applied before retrieval.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap

try:
    from scripts.evaluate_v3_matched_budget import (
        evaluate_selection, holm_adjust, indexed, prepare_grouped_artifacts,
    )
    from scripts.evaluate_v3_substitution_ablation import Engine, top_count, variant
except ModuleNotFoundError:
    from evaluate_v3_matched_budget import (
        evaluate_selection, holm_adjust, indexed, prepare_grouped_artifacts,
    )
    from evaluate_v3_substitution_ablation import Engine, top_count, variant


VARIANTS = {
    "ND0_full_nlf_utility": variant(compatibility="none", estimator="smoothed"),
    "ND1_recovery_only": variant(compatibility="none", estimator="recovery_only"),
    "ND2_unsmoothed": variant(compatibility="none", estimator="unsmoothed"),
    "ND3_source_prior_shrinkage": variant(compatibility="none", estimator="bayesian"),
    "ND4_regression_only": variant(compatibility="none", estimator="regression_only"),
}


def choose_threshold(ids, baseline, candidate, scores):
    finite = sorted({scores[key] for key in ids if math.isfinite(scores[key])})
    thresholds = [float("inf"), *finite]
    if finite:
        thresholds.append(math.nextafter(finite[0], float("-inf")))
    best = None
    for threshold in thresholds:
        selected = {key for key in ids if scores[key] > threshold}
        metrics, _ = evaluate_selection(ids, baseline, candidate, selected)
        rank = (metrics["accuracy"], metrics["macro_f1"], -len(selected), threshold)
        if best is None or rank > best[0]:
            best = (rank, threshold, metrics)
    return best[1], best[2]


def nested_calibrated_selection(ids, baseline, candidate, engine, config):
    """Choose one threshold per target fold using only the other source folds."""
    selected, diagnostics = set(), []
    folds = sorted({baseline[key]["outer_fold"] for key in ids})
    for target_fold in folds:
        calibration_ids = [key for key in ids if baseline[key]["outer_fold"] != target_fold]
        calibration_scores = {}
        for key in calibration_ids:
            calibration_scores[key], _ = engine.query_score(
                key, config, extra_excluded_folds=(target_fold,),
            )
        threshold, metrics = choose_threshold(
            calibration_ids, baseline, candidate, calibration_scores,
        )
        target_ids = [key for key in ids if baseline[key]["outer_fold"] == target_fold]
        target_scores = {key: engine.query_score(key, config)[0] for key in target_ids}
        selected.update(key for key in target_ids if target_scores[key] > threshold)
        diagnostics.append({
            "outer_fold": target_fold,
            "threshold": threshold if math.isfinite(threshold) else None,
            "calibration_n": len(calibration_ids),
            "calibration_accuracy": metrics["accuracy"],
            "calibration_macro_f1": metrics["macro_f1"],
            "calibration_route_rate": metrics["route_rate"],
        })
    return selected, diagnostics


def rank_positions(ids, scores):
    ordered = sorted(
        (key for key in ids if math.isfinite(scores[key])),
        key=lambda key: (-scores[key], key),
    )
    return {key: index + 1 for index, key in enumerate(ordered)}


def fold_analysis(ids, baseline, reference, candidate):
    rows = []
    for fold in sorted({baseline[key]["outer_fold"] for key in ids}):
        positions = [index for index, key in enumerate(ids) if baseline[key]["outer_fold"] == fold]
        gold = [baseline[ids[index]]["gold_label"] for index in positions]
        left = [reference[index] for index in positions]
        right = [candidate[index] for index in positions]
        rows.append({
            "outer_fold": fold,
            "n": len(positions),
            "reference_minus_variant_accuracy": accuracy(left, gold) - accuracy(right, gold),
            "reference_minus_variant_macro_f1": macro_f1(left, gold) - macro_f1(right, gold),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--action", default="conflicting_rescue_v1")
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument(
        "--threshold-grid", nargs="+", type=float,
        default=[-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20],
    )
    parser.add_argument("--group-threshold", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    baseline_rows = list(read_jsonl(args.baseline_records))
    memory_rows = list(read_jsonl(args.memory))
    baseline_rows, records, assignments, grouping = prepare_grouped_artifacts(
        baseline_rows, memory_rows, args.group_threshold, args.folds, args.seed,
    )
    baseline = indexed(baseline_rows, "baseline")
    candidate = indexed(read_jsonl(args.candidate), "candidate")
    examples = indexed(read_jsonl(args.examples), "examples")
    if set(baseline) != set(candidate) or set(baseline) != set(examples):
        raise ValueError("Baseline, candidate, and examples IDs must match exactly")
    ids = sorted(baseline)
    gold = [baseline[key]["gold_label"] for key in ids]
    engine = Engine(ids, baseline, records, examples, args.action, args.cost_weight)
    base_metrics, _ = evaluate_selection(ids, baseline, candidate, set())

    scores, supports = {}, {}
    for name, config in VARIANTS.items():
        print(f"VARIANT_START {name}", flush=True)
        scores[name], supports[name] = engine.score_variant(config)

    reference_scores = scores["ND0_full_nlf_utility"]
    reference_natural = {
        key for key in ids if reference_scores[key] > args.utility_threshold
    }
    matched_count = len(reference_natural)
    calibrated, calibration = nested_calibrated_selection(
        ids, baseline, candidate, engine, VARIANTS["ND0_full_nlf_utility"],
    )

    selections = {}
    for name in VARIANTS:
        selections[(name, "natural")] = {
            key for key in ids if scores[name][key] > args.utility_threshold
        }
        selections[(name, "matched_budget")] = top_count(ids, scores[name], matched_count)
    selections[("ND6_source_calibrated_threshold", "nested_natural")] = calibrated

    results, prediction_rows, route_rows, prediction_cache = {}, [], [], {}
    for (name, mode), selected in selections.items():
        metrics, predictions = evaluate_selection(ids, baseline, candidate, selected)
        metrics["delta_accuracy_vs_b1"] = metrics["accuracy"] - base_metrics["accuracy"]
        metrics["delta_macro_f1_vs_b1"] = metrics["macro_f1"] - base_metrics["macro_f1"]
        results[f"{name}:{mode}"] = metrics
        prediction_cache[(name, mode)] = predictions
        route_rows.append({"variant": name, "mode": mode, "selected_ids": sorted(selected)})
        score_map = reference_scores if name.startswith("ND6_") else scores[name]
        support_map = supports["ND0_full_nlf_utility"] if name.startswith("ND6_") else supports[name]
        for key, prediction in zip(ids, predictions):
            prediction_rows.append({
                "example_id": key, "outer_fold": assignments[key],
                "variant": name, "mode": mode, "routed": key in selected,
                "prediction": prediction, "gold_label": baseline[key]["gold_label"],
                "estimated_utility": score_map[key] if math.isfinite(score_map[key]) else None,
                "estimated_support": support_map[key],
            })

    threshold_sensitivity = []
    for threshold in args.threshold_grid:
        selected = {key for key in ids if reference_scores[key] > threshold}
        metrics, _ = evaluate_selection(ids, baseline, candidate, selected)
        threshold_sensitivity.append({"threshold": threshold, **metrics})

    reference_predictions = prediction_cache[("ND0_full_nlf_utility", "matched_budget")]
    comparisons, raw_p = {}, {}
    for name in VARIANTS:
        if name == "ND0_full_nlf_utility":
            continue
        predictions = prediction_cache[(name, "matched_budget")]
        test = mcnemar_exact(gold, predictions, reference_predictions)
        raw_p[name] = test["exact_two_sided_p"]
        comparisons[name] = {
            "mcnemar_reference_vs_variant": test,
            "paired_bootstrap_reference_minus_variant": paired_bootstrap(
                gold, predictions, reference_predictions, args.bootstrap_samples, args.seed,
            ),
            "fold_analysis": fold_analysis(ids, baseline, reference_predictions, predictions),
        }
    for name, value in holm_adjust(raw_p).items():
        comparisons[name]["holm_adjusted_accuracy_p"] = value

    smoothed_rank = rank_positions(ids, scores["ND0_full_nlf_utility"])
    unsmoothed_rank = rank_positions(ids, scores["ND2_unsmoothed"])
    utility_rows = []
    for key in ids:
        left, right = scores["ND0_full_nlf_utility"][key], scores["ND2_unsmoothed"][key]
        utility_rows.append({
            "example_id": key,
            "outer_fold": assignments[key],
            "smoothed_utility": left if math.isfinite(left) else None,
            "unsmoothed_utility": right if math.isfinite(right) else None,
            "utility_delta_smoothed_minus_unsmoothed": (
                left - right if math.isfinite(left) and math.isfinite(right) else None
            ),
            "smoothed_rank": smoothed_rank.get(key),
            "unsmoothed_rank": unsmoothed_rank.get(key),
            "rank_delta": (
                smoothed_rank[key] - unsmoothed_rank[key]
                if key in smoothed_rank and key in unsmoothed_rank else None
            ),
            "smoothed_route": key in selections[("ND0_full_nlf_utility", "natural")],
            "unsmoothed_route": key in selections[("ND2_unsmoothed", "natural")],
        })

    summary = {
        "evaluation": "grouped_fold_excluded_nlf_utility_decision_ablation",
        "configuration": {
            "action": args.action, "compatibility": "none", "top_k": 5,
            "min_support": 2, "cost_weight": args.cost_weight,
            "utility_threshold": args.utility_threshold,
            "threshold_grid": args.threshold_grid,
            "group_threshold": args.group_threshold, "folds": args.folds,
            "seed": args.seed, "matched_route_count": matched_count,
            "matched_route_rate": matched_count / len(ids),
        },
        "grouping": grouping, "baseline": base_metrics,
        "variants": VARIANTS, "results": results,
        "threshold_sensitivity": threshold_sensitivity,
        "source_calibration": calibration,
        "matched_budget_reference_comparisons": comparisons,
        "nd2_diagnostics": {
            "same_natural_routes": sum(
                row["smoothed_route"] == row["unsmoothed_route"] for row in utility_rows
            ),
            "different_natural_routes": sum(
                row["smoothed_route"] != row["unsmoothed_route"] for row in utility_rows
            ),
            "same_rank": sum(row["rank_delta"] == 0 for row in utility_rows),
            "max_absolute_rank_delta": max(
                abs(row["rank_delta"]) for row in utility_rows if row["rank_delta"] is not None
            ),
        },
        "protocol_notes": [
            "ND3 uses empirical recovery/regression priors from source records remaining after target-fold exclusion, with prior strength 2.",
            "ND6 calibrates one threshold per target fold using only the other source folds; each calibration query also excludes its own grouped fold.",
            "ND5 thresholds are predeclared and are not selected on external datasets.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(args.output_dir / "routes.jsonl", route_rows)
    write_jsonl(args.output_dir / "nd2_utility_diagnostics.jsonl", utility_rows)
    print(json.dumps({
        "matched_route_count": matched_count,
        "variants": len(VARIANTS),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
