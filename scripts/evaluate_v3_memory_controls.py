"""Evaluate extended V3 outcome-memory controls from cached B1/A3 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_memory import ResidualMemoryStore
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap

try:
    from scripts.evaluate_v3_component_ablation import route_overlap, top_count
    from scripts.evaluate_v3_matched_budget import (
        evaluate_selection,
        holm_adjust,
        indexed,
        prepare_grouped_artifacts,
    )
    from scripts.evaluate_v3_memory_ablation import (
        FULL_WEIGHTS,
        outcome_counts,
        population_records,
        shuffle_outcomes,
    )
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from evaluate_v3_component_ablation import route_overlap, top_count
    from evaluate_v3_matched_budget import (
        evaluate_selection,
        holm_adjust,
        indexed,
        prepare_grouped_artifacts,
    )
    from evaluate_v3_memory_ablation import (
        FULL_WEIGHTS,
        outcome_counts,
        population_records,
        shuffle_outcomes,
    )


PRIMARY_VARIANTS = (
    "B10_recovery_only",
    "B11_regression_only",
    "B12_size_matched_full",
    "B13_size_matched_mixed_outcome",
    "B14_stratified_outcome_shuffle",
    "B15_class_conditional",
    "B16_uniform_outcome_weights",
    "B17_calibrated_utility",
)
STOCHASTIC_VARIANTS = {
    "B12_size_matched_full",
    "B13_size_matched_mixed_outcome",
    "B14_stratified_outcome_shuffle",
}
BUDGET_VARIANTS = (
    "B7_error_only",
    "B9_full_outcome_memory",
    "B10_recovery_only",
    "B14_stratified_outcome_shuffle",
    "confidence_only",
)


def stable_sample(records, count, seed, namespace="uniform"):
    """Select a reproducible subset without depending on input ordering."""
    if count < 0 or count > len(records):
        raise ValueError(f"sample count {count} is invalid for {len(records)} records")
    ranked = sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{namespace}:{seed}:{record.memory_id}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[:count]


def outcome_bucket(record):
    if record.recovery_count:
        return "recovery"
    if record.regression_count:
        return "regression"
    return "neutral"


def stratified_outcome_sample(records, count, seed):
    """Sample exactly ``count`` records while approximating full outcome shares."""
    groups = defaultdict(list)
    for record in records:
        groups[outcome_bucket(record)].append(record)
    if count > len(records):
        raise ValueError(f"sample count {count} exceeds {len(records)} records")
    exact = {key: count * len(rows) / len(records) for key, rows in groups.items()}
    allocation = {key: min(len(groups[key]), math.floor(value)) for key, value in exact.items()}
    remaining = count - sum(allocation.values())
    order = sorted(groups, key=lambda key: (-(exact[key] - allocation[key]), key))
    while remaining:
        progressed = False
        for key in order:
            if allocation[key] < len(groups[key]):
                allocation[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise ValueError("could not satisfy stratified sample size")
    output = []
    for key in sorted(groups):
        output.extend(stable_sample(groups[key], allocation[key], seed, f"outcome={key}"))
    return sorted(output, key=lambda record: record.memory_id)


def estimate_neighbors(neighbors, action, cost_weight, utility_mode="full",
                       uniform_outcome_weights=False, smoothing=1.0):
    rows = [(record, 1.0 if uniform_outcome_weights else similarity)
            for record, similarity in neighbors if record.correction_strategy == action]
    if not rows:
        return float("-inf"), 0, False, {"recovery": 0.0, "regression": 0.0, "cost": 0.0}
    weight = sum(similarity for _, similarity in rows)
    fallback = weight <= 0.0
    if fallback:
        rows = [(record, 1.0) for record, _ in rows]
        weight = float(len(rows))
    denominator = weight + 2.0 * smoothing
    recovery = (sum(record.recovery_count * value for record, value in rows) + smoothing) / denominator
    regression = (sum(record.regression_count * value for record, value in rows) + smoothing) / denominator
    cost = sum((record.estimated_extra_cost or 0.0) * value for record, value in rows) / weight
    if utility_mode == "full":
        utility = recovery - regression - cost_weight * cost
    elif utility_mode == "recovery_only":
        utility = recovery - cost_weight * cost
    elif utility_mode == "regression_only":
        # Safety score: larger means lower estimated regression risk.
        utility = 1.0 - regression - cost_weight * cost
    else:
        raise ValueError(f"unknown utility mode: {utility_mode}")
    return utility, len(rows), fallback, {
        "recovery": recovery, "regression": regression, "cost": cost,
    }


def memory_scores(ids, baseline, records, action, top_k, min_support, cost_weight,
                  utility_mode="full", uniform_outcome_weights=False,
                  compatibility_mode="same_label"):
    store = ResidualMemoryStore(records)
    scores, supports, fallbacks, estimates = {}, {}, {}, {}
    for key in ids:
        row = baseline[key]
        query = {
            "example_id": key,
            "claim": row["claim"],
            "retrieval_features": {
                "baseline_prediction": row["prediction"],
                "label_scores": row.get("label_scores", {}),
                "evidence_tokens": row.get("evidence_tokens", 0),
            },
        }
        neighbors = store.retrieve(
            query,
            top_k=top_k,
            action=action,
            exclude_claim_ids={key},
            exclude_folds={row["outer_fold"]},
            similarity_weights=FULL_WEIGHTS,
            compatibility_mode=compatibility_mode,
        )
        utility, support, fallback, estimate = estimate_neighbors(
            neighbors, action, cost_weight, utility_mode, uniform_outcome_weights,
        )
        scores[key] = utility if support >= min_support else float("-inf")
        supports[key] = support
        fallbacks[key] = fallback
        estimates[key] = estimate
    return scores, supports, fallbacks, estimates


def class_conditional_scores(ids, baseline, records, action, min_support, cost_weight):
    scores, supports, fallbacks, estimates = {}, {}, {}, {}
    for key in ids:
        row = baseline[key]
        neighbors = [
            (record, 1.0) for record in records
            if record.active
            and record.source_split == "train"
            and record.correction_strategy == action
            and record.source_claim_id != key
            and record.provenance.get("outer_fold") != row["outer_fold"]
            and record.retrieval_features.get("baseline_prediction") == row["prediction"]
        ]
        utility, support, fallback, estimate = estimate_neighbors(
            neighbors, action, cost_weight, "full", False,
        )
        scores[key] = utility if support >= min_support else float("-inf")
        supports[key] = support
        fallbacks[key] = fallback
        estimates[key] = estimate
    return scores, supports, fallbacks, estimates


def confidence_scores(ids, baseline):
    result = {}
    for key in ids:
        row = baseline[key]
        probabilities = row.get("label_probabilities", {})
        result[key] = (
            1.0 - max(probabilities.values())
            if probabilities else -float(row.get("prediction_margin") or 0.0)
        )
    return result


def threshold_candidates(scores):
    finite = sorted({value for value in scores.values() if math.isfinite(value)})
    if not finite:
        return [float("inf")]
    return [float("inf"), *finite, math.nextafter(finite[0], float("-inf"))]


def choose_threshold(ids, baseline, candidate, scores):
    """Maximize source-inner accuracy, then F1, then prefer fewer routes."""
    best = None
    for threshold in threshold_candidates(scores):
        selected = {key for key in ids if scores[key] > threshold}
        metrics, _ = evaluate_selection(ids, baseline, candidate, selected)
        rank = (metrics["accuracy"], metrics["macro_f1"], -len(selected), threshold)
        if best is None or rank > best[0]:
            best = (rank, threshold, metrics)
    return best[1], best[2]


def calibrated_selection(ids, baseline, candidate, records, action, top_k,
                         min_support, cost_weight, compatibility_mode="same_label"):
    folds = sorted({baseline[key]["outer_fold"] for key in ids})
    selected, diagnostics = set(), []
    for target_fold in folds:
        source_records = [
            record for record in records
            if record.provenance.get("outer_fold") != target_fold
        ]
        calibration_ids = [key for key in ids if baseline[key]["outer_fold"] != target_fold]
        scores, _, _, _ = memory_scores(
            calibration_ids, baseline, source_records, action, top_k,
            min_support, cost_weight, compatibility_mode=compatibility_mode,
        )
        threshold, calibration_metrics = choose_threshold(
            calibration_ids, baseline, candidate, scores,
        )
        target_ids = [key for key in ids if baseline[key]["outer_fold"] == target_fold]
        target_scores, _, _, _ = memory_scores(
            target_ids, baseline, records, action, top_k, min_support, cost_weight,
            compatibility_mode=compatibility_mode,
        )
        selected.update(key for key in target_ids if target_scores[key] > threshold)
        diagnostics.append({
            "outer_fold": target_fold,
            "threshold": threshold if math.isfinite(threshold) else None,
            "calibration_n": len(calibration_ids),
            "calibration_accuracy": calibration_metrics["accuracy"],
            "calibration_macro_f1": calibration_metrics["macro_f1"],
            "calibration_route_rate": calibration_metrics["route_rate"],
        })
    return selected, diagnostics


def summarize_seed_rows(rows):
    keys = ("routed", "route_rate", "accuracy", "macro_f1", "helpful", "harmful", "net")
    output = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        output[key] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return output


def comparison(gold, reference_predictions, variant_predictions, samples, seed,
               ids=None, assignments=None):
    report = {
        "mcnemar_exact": mcnemar_exact(gold, variant_predictions, reference_predictions),
        "paired_bootstrap_reference_minus_variant": paired_bootstrap(
            gold, variant_predictions, reference_predictions, samples, seed,
        ),
        "reference_minus_variant_accuracy": (
            accuracy(reference_predictions, gold) - accuracy(variant_predictions, gold)
        ),
        "reference_minus_variant_macro_f1": (
            macro_f1(reference_predictions, gold) - macro_f1(variant_predictions, gold)
        ),
    }
    if ids is not None and assignments is not None:
        folds = []
        for fold in sorted(set(assignments.values())):
            positions = [index for index, key in enumerate(ids) if assignments[key] == fold]
            fold_gold = [gold[index] for index in positions]
            fold_reference = [reference_predictions[index] for index in positions]
            fold_variant = [variant_predictions[index] for index in positions]
            folds.append({
                "outer_fold": fold,
                "n": len(positions),
                "reference_accuracy": accuracy(fold_reference, fold_gold),
                "variant_accuracy": accuracy(fold_variant, fold_gold),
                "reference_minus_variant_accuracy": (
                    accuracy(fold_reference, fold_gold) - accuracy(fold_variant, fold_gold)
                ),
                "reference_macro_f1": macro_f1(fold_reference, fold_gold),
                "variant_macro_f1": macro_f1(fold_variant, fold_gold),
                "reference_minus_variant_macro_f1": (
                    macro_f1(fold_reference, fold_gold) - macro_f1(fold_variant, fold_gold)
                ),
            })
        report["outer_fold_analysis"] = {
            "folds": folds,
            "reference_better_accuracy_folds": sum(
                row["reference_minus_variant_accuracy"] > 0 for row in folds
            ),
            "reference_better_macro_f1_folds": sum(
                row["reference_minus_variant_macro_f1"] > 0 for row in folds
            ),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--action", default="conflicting_rescue_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument(
        "--compatibility-mode", choices=["same_label", "none"],
        default="same_label",
        help="Use 'none' for final R6-NLF; default reproduces historical HardLabel controls.",
    )
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--group-threshold", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=list(range(2026, 2046)))
    parser.add_argument("--size-match-count", type=int, default=None)
    parser.add_argument("--budgets", nargs="+", type=float,
                        default=[0.05, 0.10, 0.1118, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    if any(not 0.0 <= value <= 1.0 for value in args.budgets):
        raise ValueError("budgets must lie in [0, 1]")
    if args.seed not in args.random_seeds:
        raise ValueError("the primary --seed must also appear in --random-seeds")

    baseline_rows = list(read_jsonl(args.baseline_records))
    candidate_rows = list(read_jsonl(args.candidate))
    memory_rows = list(read_jsonl(args.memory))
    baseline_rows, records, assignments, grouping = prepare_grouped_artifacts(
        baseline_rows, memory_rows, args.group_threshold, args.folds, args.seed,
    )
    baseline = indexed(baseline_rows, "baseline")
    candidate = indexed(candidate_rows, "candidate")
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate IDs must match exactly")
    ids = sorted(baseline)
    gold = [baseline[key]["gold_label"] for key in ids]
    base_metrics, base_predictions = evaluate_selection(ids, baseline, candidate, set())
    error_records = population_records(records, "error_only")
    size_match_count = len(error_records) if args.size_match_count is None else args.size_match_count
    if size_match_count != len(error_records):
        print(
            f"MEMORY_CONTROL_WARNING requested_size={size_match_count} "
            f"error_memory_size={len(error_records)}",
            flush=True,
        )

    print("MEMORY_CONTROLS_COMPUTE reference=B9", flush=True)
    b9_scores = memory_scores(
        ids, baseline, records, args.action, args.top_k, args.min_support,
        args.cost_weight, compatibility_mode=args.compatibility_mode,
    )
    b9_natural = {key for key in ids if b9_scores[0][key] > args.utility_threshold}
    matched_count = len(b9_natural)
    results, routes, prediction_rows = {}, [], []
    prediction_index = {}

    def record_run(variant, mode, seed, selected, scores, supports, fallbacks,
                   estimates, save=True):
        metrics, predictions = evaluate_selection(ids, baseline, candidate, selected)
        metrics["delta_accuracy_vs_b1"] = metrics["accuracy"] - base_metrics["accuracy"]
        metrics["delta_macro_f1_vs_b1"] = metrics["macro_f1"] - base_metrics["macro_f1"]
        metrics["route_overlap_with_b9_natural"] = route_overlap(selected, b9_natural)
        metrics["seed"] = seed
        key = f"{variant}:{mode}" + (f":seed={seed}" if seed is not None else "")
        results[key] = metrics
        routes.append({"variant": variant, "mode": mode, "seed": seed,
                       "selected_ids": sorted(selected)})
        prediction_index[(variant, mode, seed)] = predictions
        if save:
            for example_id, prediction in zip(ids, predictions):
                estimate = estimates.get(example_id, {})
                prediction_rows.append({
                    "example_id": example_id,
                    "outer_fold": assignments[example_id],
                    "variant": variant,
                    "mode": mode,
                    "seed": seed,
                    "routed": example_id in selected,
                    "prediction": prediction,
                    "gold_label": baseline[example_id]["gold_label"],
                    "estimated_utility": (
                        scores.get(example_id)
                        if math.isfinite(scores.get(example_id, float("-inf"))) else None
                    ),
                    "estimated_support": supports.get(example_id, 0),
                    "estimated_recovery": estimate.get("recovery"),
                    "estimated_regression": estimate.get("regression"),
                    "uniform_weight_fallback": fallbacks.get(example_id, False),
                })
        return metrics, predictions

    deterministic = {"B9_full_outcome_memory": b9_scores}
    deterministic_jobs = [
        ("B10_recovery_only", lambda: memory_scores(
            ids, baseline, records, args.action, args.top_k, args.min_support,
            args.cost_weight, utility_mode="recovery_only",
            compatibility_mode=args.compatibility_mode,
        )),
        ("B11_regression_only", lambda: memory_scores(
            ids, baseline, records, args.action, args.top_k, args.min_support,
            args.cost_weight, utility_mode="regression_only",
            compatibility_mode=args.compatibility_mode,
        )),
        ("B15_class_conditional", lambda: class_conditional_scores(
            ids, baseline, records, args.action, args.min_support, args.cost_weight,
        )),
        ("B16_uniform_outcome_weights", lambda: memory_scores(
            ids, baseline, records, args.action, args.top_k, args.min_support,
            args.cost_weight, uniform_outcome_weights=True,
            compatibility_mode=args.compatibility_mode,
        )),
    ]
    for index, (variant, job) in enumerate(deterministic_jobs, 1):
        print(
            f"MEMORY_CONTROLS_COMPUTE deterministic={variant} "
            f"step={index}/{len(deterministic_jobs)}",
            flush=True,
        )
        deterministic[variant] = job()
    for variant, bundle in deterministic.items():
        scores, supports, fallbacks, estimates = bundle
        natural = {key for key in ids if scores[key] > args.utility_threshold}
        matched = top_count(ids, scores, matched_count)
        record_run(variant, "natural", None, natural, *bundle)
        record_run(variant, "matched_budget", None, matched, *bundle)

    b7_bundle = memory_scores(
        ids, baseline, error_records, args.action, args.top_k, args.min_support,
        args.cost_weight,
        compatibility_mode=args.compatibility_mode,
    )
    stochastic_rows = {variant: {"natural": [], "matched_budget": []}
                       for variant in STOCHASTIC_VARIANTS}
    stochastic_memory_diagnostics = []
    primary_stochastic = {}
    started = time.perf_counter()
    for index, seed in enumerate(args.random_seeds, 1):
        sampled_full = stable_sample(records, size_match_count, seed)
        sampled_mixed = stratified_outcome_sample(records, size_match_count, seed)
        shuffled = shuffle_outcomes(records, seed)
        stochastic_memory_diagnostics.append({
            "seed": seed,
            "B12_size_matched_full": outcome_counts(sampled_full),
            "B13_size_matched_mixed_outcome": outcome_counts(sampled_mixed),
            "B14_stratified_outcome_shuffle": outcome_counts(shuffled),
        })
        bundles = {
            "B12_size_matched_full": memory_scores(
                ids, baseline, sampled_full, args.action, args.top_k,
                args.min_support, args.cost_weight,
                compatibility_mode=args.compatibility_mode,
            ),
            "B13_size_matched_mixed_outcome": memory_scores(
                ids, baseline, sampled_mixed, args.action, args.top_k,
                args.min_support, args.cost_weight,
                compatibility_mode=args.compatibility_mode,
            ),
            "B14_stratified_outcome_shuffle": memory_scores(
                ids, baseline, shuffled, args.action, args.top_k,
                args.min_support, args.cost_weight,
                compatibility_mode=args.compatibility_mode,
            ),
        }
        for variant, bundle in bundles.items():
            scores, supports, fallbacks, estimates = bundle
            if seed == args.seed:
                primary_stochastic[variant] = bundle
            for mode, selected in (
                ("natural", {key for key in ids if scores[key] > args.utility_threshold}),
                ("matched_budget", top_count(ids, scores, matched_count)),
            ):
                metrics, _ = record_run(
                    variant, mode, seed, selected, *bundle, save=True,
                )
                stochastic_rows[variant][mode].append(metrics)
        elapsed = time.perf_counter() - started
        rate = index / elapsed if elapsed else 0.0
        eta = (len(args.random_seeds) - index) / rate if rate else 0.0
        print(
            f"MEMORY_CONTROLS_PROGRESS seeds={index}/{len(args.random_seeds)} "
            f"elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
            flush=True,
        )

    print("MEMORY_CONTROLS_COMPUTE calibrated=B17", flush=True)
    b17_selected, calibration = calibrated_selection(
        ids, baseline, candidate, records, args.action, args.top_k,
        args.min_support, args.cost_weight, args.compatibility_mode,
    )
    b17_metrics, b17_predictions = record_run(
        "B17_calibrated_utility", "nested_calibrated", None, b17_selected,
        *b9_scores,
    )

    primary_predictions = {
        "B9_full_outcome_memory": prediction_index[("B9_full_outcome_memory", "matched_budget", None)],
        "B10_recovery_only": prediction_index[("B10_recovery_only", "matched_budget", None)],
        "B11_regression_only": prediction_index[("B11_regression_only", "matched_budget", None)],
        "B15_class_conditional": prediction_index[("B15_class_conditional", "matched_budget", None)],
        "B16_uniform_outcome_weights": prediction_index[("B16_uniform_outcome_weights", "matched_budget", None)],
        "B17_calibrated_utility": b17_predictions,
    }
    for variant in STOCHASTIC_VARIANTS:
        primary_predictions[variant] = prediction_index[(variant, "matched_budget", args.seed)]
    reference = primary_predictions["B9_full_outcome_memory"]
    primary_comparisons, raw_p = {}, {}
    for variant in PRIMARY_VARIANTS:
        print(f"MEMORY_CONTROLS_SIGNIFICANCE primary={variant}", flush=True)
        report = comparison(
            gold, reference, primary_predictions[variant],
            args.bootstrap_samples, args.seed, ids, assignments,
        )
        primary_comparisons[variant] = report
        raw_p[variant] = report["mcnemar_exact"]["exact_two_sided_p"]
    adjusted = holm_adjust(raw_p)
    for variant, value in adjusted.items():
        primary_comparisons[variant]["holm_adjusted_mcnemar_p"] = value

    budget_results, budget_predictions, budget_comparisons = {}, [], {}
    budget_bundles = {
        "B7_error_only": b7_bundle,
        "B9_full_outcome_memory": b9_scores,
        "B10_recovery_only": deterministic["B10_recovery_only"],
        "B14_stratified_outcome_shuffle": primary_stochastic["B14_stratified_outcome_shuffle"],
    }
    conf = confidence_scores(ids, baseline)
    budget_bundles["confidence_only"] = (
        conf, {key: 0 for key in ids}, {key: False for key in ids},
        {key: {} for key in ids},
    )
    for budget in args.budgets:
        print(f"MEMORY_CONTROLS_BUDGET budget={budget:.4f}", flush=True)
        count = round(len(ids) * budget)
        predictions_at_budget = {}
        for variant in BUDGET_VARIANTS:
            bundle = budget_bundles[variant]
            selected = top_count(ids, bundle[0], count)
            metrics, predictions = evaluate_selection(ids, baseline, candidate, selected)
            result_key = f"{variant}:budget={budget:.4f}"
            budget_results[result_key] = {"budget": budget, **metrics}
            predictions_at_budget[variant] = predictions
            for example_id, prediction in zip(ids, predictions):
                budget_predictions.append({
                    "example_id": example_id,
                    "outer_fold": assignments[example_id],
                    "variant": variant,
                    "budget": budget,
                    "routed": example_id in selected,
                    "prediction": prediction,
                    "gold_label": baseline[example_id]["gold_label"],
                })
        budget_reference = predictions_at_budget["B9_full_outcome_memory"]
        family, family_p = {}, {}
        for variant in BUDGET_VARIANTS:
            if variant == "B9_full_outcome_memory":
                continue
            print(
                f"MEMORY_CONTROLS_SIGNIFICANCE budget={budget:.4f} "
                f"control={variant}",
                flush=True,
            )
            report = comparison(
                gold, budget_reference, predictions_at_budget[variant],
                args.bootstrap_samples, args.seed, ids, assignments,
            )
            family[variant] = report
            family_p[variant] = report["mcnemar_exact"]["exact_two_sided_p"]
        family_adjusted = holm_adjust(family_p)
        for variant, value in family_adjusted.items():
            family[variant]["holm_adjusted_mcnemar_p"] = value
        budget_comparisons[f"{budget:.4f}"] = family

    stochastic_aggregate = {
        variant: {mode: summarize_seed_rows(rows) for mode, rows in modes.items()}
        for variant, modes in stochastic_rows.items()
    }
    report = {
        "evaluation": "grouped_fold_excluded_extended_memory_controls",
        "configuration": {
            "action": args.action,
            "top_k": args.top_k,
            "min_support": args.min_support,
            "cost_weight": args.cost_weight,
            "utility_threshold": args.utility_threshold,
            "group_threshold": args.group_threshold,
            "folds": args.folds,
            "seed": args.seed,
            "random_seeds": args.random_seeds,
            "size_match_count": size_match_count,
            "error_memory_observed_count": len(error_records),
            "matched_route_count": matched_count,
            "compatibility_mode": args.compatibility_mode,
            "matched_route_rate": matched_count / len(ids),
            "budgets": args.budgets,
            "b11_score": "1 - estimated_regression - cost_weight * estimated_cost",
            "b17_selection": "nested source-inner threshold maximizing accuracy, then macro-F1, then fewer routes",
        },
        "grouping": grouping,
        "baseline": base_metrics,
        "memory_populations": {
            "full": outcome_counts(records),
            "error_only": outcome_counts(error_records),
        },
        "results": results,
        "stochastic_aggregate": stochastic_aggregate,
        "stochastic_memory_diagnostics": stochastic_memory_diagnostics,
        "primary_matched_comparisons_b9_minus_variant": primary_comparisons,
        "b17_calibration": calibration,
        "budget_results": budget_results,
        "budget_comparisons_b9_minus_control": budget_comparisons,
        "integrity_notes": [
            "B14 is the confirmatory name for the existing fold-and-label-stratified B4 shuffle.",
            "B15 is mathematically identical to B5 under the frozen class-conditional definition.",
            "B12 samples uniformly from full memory at the B7/error-memory size.",
            "B13 samples recovery, regression, and neutral outcomes proportionally at the same size.",
            "All thresholds and routing decisions are computed without external evaluation labels.",
            "Compatibility mode is explicit; 'none' denotes the final R6-NLF router.",
            "Holm correction is applied to one B10--B17 family and separately within each budget family.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", report)
    write_jsonl(args.output_dir / "routes.jsonl", routes)
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(args.output_dir / "budget_predictions.jsonl", budget_predictions)
    print(json.dumps({
        "matched_route_count": matched_count,
        "size_match_count": size_match_count,
        "primary_results": {
            key: value for key, value in results.items()
            if ("seed=" not in key or key.endswith(f"seed={args.seed}"))
        },
        "stochastic_aggregate": stochastic_aggregate,
        "primary_comparisons": primary_comparisons,
        "b17": b17_metrics,
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
