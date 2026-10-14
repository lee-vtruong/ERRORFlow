"""Evaluate outcome-memory ablations from cached AVeriTeC B1/A3 artifacts."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_memory import ResidualMemoryStore
from errorflow.v3_routing import best_memory_action, estimate_action

try:
    from scripts.evaluate_v3_component_ablation import (
        evaluate_variant,
        route_overlap,
        summarize_replicates,
        top_count,
        utility_scores,
    )
    from scripts.evaluate_v3_matched_budget import evaluate_selection, indexed, prepare_grouped_artifacts
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from evaluate_v3_component_ablation import (
        evaluate_variant,
        route_overlap,
        summarize_replicates,
        top_count,
        utility_scores,
    )
    from evaluate_v3_matched_budget import evaluate_selection, indexed, prepare_grouped_artifacts


FULL_WEIGHTS = (0.35, 0.50, 0.15)
VARIANTS = {
    "B0_no_memory": "B1 prediction; no A3 invocation",
    "B1_random_memory": "random compatible top-k records",
    "B2_claim_nearest": "claim-similarity top-k",
    "B3_score_nearest": "label-score-similarity top-k",
    "B4_outcome_shuffled": "full retrieval after within-fold/label outcome permutation",
    "B5_static_class": "uniform aggregate by B1 predicted label",
    "B6_static_global": "uniform global aggregate",
    "B7_error_only": "full retrieval over source examples where B1 was wrong",
    "B8_correct_only": "full retrieval over source examples where B1 was correct",
    "B9_full_outcome_memory": "full R6 outcome memory",
}


def shuffle_outcomes(records, seed):
    """Break feature--outcome association while preserving fold/label marginals."""
    groups = defaultdict(list)
    for record in records:
        key = (
            record.provenance.get("outer_fold"),
            record.retrieval_features.get("baseline_prediction"),
        )
        groups[key].append(record)
    output = []
    for key in sorted(groups, key=lambda item: (str(item[0]), str(item[1]))):
        members = sorted(groups[key], key=lambda record: record.memory_id)
        outcomes = [
            (
                record.recovery_count,
                record.regression_count,
                record.estimated_net_utility,
                record.memory_id,
            )
            for record in members
        ]
        group_rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
        group_rng.shuffle(outcomes)
        for record, (recovery, regression, utility, source_memory_id) in zip(members, outcomes):
            provenance = dict(record.provenance)
            provenance["outcome_shuffle"] = {
                "seed": seed,
                "stratum": [key[0], key[1]],
                "source_memory_id": source_memory_id,
            }
            output.append(replace(
                record,
                recovery_count=recovery,
                regression_count=regression,
                estimated_net_utility=utility,
                provenance=provenance,
            ))
    return output


def population_records(records, population):
    if population == "full":
        return list(records)
    expected_correct = population == "correct_only"
    return [
        record for record in records
        if bool(record.provenance.get("baseline_correct_offline_only")) == expected_correct
    ]


def static_utility_scores(ids, baseline, records, action, min_support, cost_weight,
                          class_level):
    scores, supports, fallbacks = {}, {}, {}
    for key in ids:
        row = baseline[key]
        neighbors = []
        for record in records:
            if not record.active or record.source_split != "train":
                continue
            if record.correction_strategy != action:
                continue
            if record.source_claim_id == key:
                continue
            if record.provenance.get("outer_fold") == row["outer_fold"]:
                continue
            if class_level and record.retrieval_features.get("baseline_prediction") != row["prediction"]:
                continue
            neighbors.append((record, 1.0))
        estimate = estimate_action(action, neighbors, cost_weight)
        scores[key] = estimate.utility if estimate.support >= min_support else float("-inf")
        supports[key] = estimate.support
        fallbacks[key] = estimate.uniform_weight_fallback
    return scores, supports, fallbacks


def outcome_counts(records):
    return {
        "records": len(records),
        "recovery": sum(record.recovery_count for record in records),
        "regression": sum(record.regression_count for record in records),
        "neutral": sum(
            record.recovery_count == 0 and record.regression_count == 0 for record in records
        ),
    }


def summarize_memory_replicates(rows):
    compatible = []
    for row in rows:
        copied = dict(row)
        copied["route_overlap_with_a0_natural"] = row["route_overlap_with_b9_natural"]
        compatible.append(copied)
    summary = summarize_replicates(compatible)
    summary["route_jaccard_with_b9_natural"] = summary.pop(
        "route_jaccard_with_a0_natural"
    )
    return summary


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
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--group-threshold", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=list(range(2026, 2046)))
    args = parser.parse_args()

    baseline_rows = list(read_jsonl(args.baseline_records))
    candidate_rows = list(read_jsonl(args.candidate))
    memory_rows = list(read_jsonl(args.memory))
    baseline_rows, records, _, grouping = prepare_grouped_artifacts(
        baseline_rows, memory_rows, args.group_threshold, args.folds, args.seed,
    )
    baseline = indexed(baseline_rows, "baseline")
    candidate = indexed(candidate_rows, "candidate")
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate IDs must match exactly")
    ids = sorted(baseline)
    base_metrics, _ = evaluate_selection(ids, baseline, candidate, set())

    full_scores, full_supports, full_fallbacks = utility_scores(
        ids, baseline, records, args.action, args.top_k, args.min_support,
        args.cost_weight, FULL_WEIGHTS, "ranked", args.seed,
    )
    full_natural = {key for key in ids if full_scores[key] > args.utility_threshold}
    matched_count = len(full_natural)

    results, routes, prediction_rows = {}, [], []

    def record_run(variant, mode, seed, selected, scores, supports, fallbacks,
                   save_predictions=True):
        metrics, predictions = evaluate_variant(ids, baseline, candidate, selected, base_metrics)
        metrics["route_overlap_with_b9_natural"] = route_overlap(selected, full_natural)
        metrics["uniform_weight_fallback_count"] = sum(fallbacks.values())
        metrics["uniform_weight_fallback_rate"] = sum(fallbacks.values()) / len(ids)
        metrics["seed"] = seed
        result_key = f"{variant}:{mode}" + (f":seed={seed}" if seed is not None else "")
        results[result_key] = metrics
        routes.append({
            "variant": variant, "mode": mode, "seed": seed,
            "selected_ids": sorted(selected),
        })
        if save_predictions:
            for key, prediction in zip(ids, predictions):
                prediction_rows.append({
                    "example_id": key, "variant": variant, "mode": mode,
                    "seed": seed, "routed": key in selected,
                    "prediction": prediction, "gold_label": baseline[key]["gold_label"],
                    "estimated_utility": (
                        scores.get(key) if math.isfinite(scores.get(key, float("-inf"))) else None
                    ),
                    "estimated_support": supports.get(key, 0),
                })
        return metrics

    zero_scores = {key: float("-inf") for key in ids}
    zero_supports = {key: 0 for key in ids}
    zero_fallbacks = {key: False for key in ids}
    for mode in ("natural", "matched_budget"):
        record_run(
            "B0_no_memory", mode, None, set(), zero_scores, zero_supports,
            zero_fallbacks,
        )

    deterministic = {}
    deterministic_jobs = [
        ("B2_claim_nearest", lambda: utility_scores(
            ids, baseline, records, args.action, args.top_k, args.min_support,
            args.cost_weight, (1.0, 0.0, 0.0), "ranked", args.seed,
        )),
        ("B3_score_nearest", lambda: utility_scores(
            ids, baseline, records, args.action, args.top_k, args.min_support,
            args.cost_weight, (0.0, 1.0, 0.0), "ranked", args.seed,
        )),
        ("B5_static_class", lambda: static_utility_scores(
            ids, baseline, records, args.action, args.min_support,
            args.cost_weight, class_level=True,
        )),
        ("B6_static_global", lambda: static_utility_scores(
            ids, baseline, records, args.action, args.min_support,
            args.cost_weight, class_level=False,
        )),
        ("B7_error_only", lambda: utility_scores(
            ids, baseline, population_records(records, "error_only"), args.action,
            args.top_k, args.min_support, args.cost_weight, FULL_WEIGHTS,
            "ranked", args.seed,
        )),
        ("B8_correct_only", lambda: utility_scores(
            ids, baseline, population_records(records, "correct_only"), args.action,
            args.top_k, args.min_support, args.cost_weight, FULL_WEIGHTS,
            "ranked", args.seed,
        )),
    ]
    for index, (variant, job) in enumerate(deterministic_jobs, 1):
        print(f"MEMORY_ABLATION_COMPUTE deterministic={variant} step={index}/{len(deterministic_jobs)}", flush=True)
        deterministic[variant] = job()
    deterministic["B9_full_outcome_memory"] = (full_scores, full_supports, full_fallbacks)
    for variant, (scores, supports, fallbacks) in deterministic.items():
        natural = {key for key in ids if scores[key] > args.utility_threshold}
        matched = top_count(ids, scores, matched_count)
        record_run(variant, "natural", None, natural, scores, supports, fallbacks)
        record_run(variant, "matched_budget", None, matched, scores, supports, fallbacks)

    stochastic = {"B1_random_memory": {"natural": [], "matched_budget": []},
                  "B4_outcome_shuffled": {"natural": [], "matched_budget": []}}
    stochastic_started = time.perf_counter()
    for seed_index, seed in enumerate(args.random_seeds, 1):
        print(
            f"MEMORY_ABLATION_COMPUTE stochastic_seed={seed} "
            f"step={seed_index}/{len(args.random_seeds)}",
            flush=True,
        )
        random_scores = utility_scores(
            ids, baseline, records, args.action, args.top_k, args.min_support,
            args.cost_weight, FULL_WEIGHTS, "random", seed,
        )
        shuffled_records = shuffle_outcomes(records, seed)
        shuffled_scores = utility_scores(
            ids, baseline, shuffled_records, args.action, args.top_k,
            args.min_support, args.cost_weight, FULL_WEIGHTS, "ranked", seed,
        )
        for variant, (scores, supports, fallbacks) in (
            ("B1_random_memory", random_scores),
            ("B4_outcome_shuffled", shuffled_scores),
        ):
            for mode, selected in (
                ("natural", {key for key in ids if scores[key] > args.utility_threshold}),
                ("matched_budget", top_count(ids, scores, matched_count)),
            ):
                metrics = record_run(
                    variant, mode, seed, selected, scores, supports, fallbacks,
                    save_predictions=seed == args.seed,
                )
                stochastic[variant][mode].append(metrics)
        elapsed = time.perf_counter() - stochastic_started
        rate = seed_index / elapsed if elapsed else 0.0
        eta = (len(args.random_seeds) - seed_index) / rate if rate else 0.0
        print(
            f"MEMORY_ABLATION_PROGRESS seeds={seed_index}/{len(args.random_seeds)} "
            f"elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
            flush=True,
        )

    stochastic_aggregate = {
        variant: {mode: summarize_memory_replicates(rows) for mode, rows in modes.items()}
        for variant, modes in stochastic.items()
    }
    primary_results = {
        key: value for key, value in results.items()
        if "seed=" not in key or key.endswith(f"seed={args.seed}")
    }
    report = {
        "evaluation": "grouped_fold_excluded_outcome_memory_ablation",
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
            "matched_route_count": matched_count,
            "matched_route_rate": matched_count / len(ids),
            "outcome_shuffle_strata": ["outer_fold", "baseline_prediction"],
        },
        "grouping": grouping,
        "variants": VARIANTS,
        "memory_populations": {
            "full": outcome_counts(records),
            "error_only": outcome_counts(population_records(records, "error_only")),
            "correct_only": outcome_counts(population_records(records, "correct_only")),
        },
        "baseline": base_metrics,
        "primary_results": primary_results,
        "all_results": results,
        "stochastic_aggregate": stochastic_aggregate,
        "limitations": [
            "Development-fold ablation only; no external target result is used for selection.",
            "B7/B8 deliberately alter the memory population and therefore expose selection bias.",
            "B6 has no individual ranking; matched-budget ties are resolved deterministically by example ID.",
            "B1 and B4 use seed 2026 as primary and 20 seeds for sensitivity.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", report)
    write_jsonl(args.output_dir / "routes.jsonl", routes)
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)
    print(json.dumps({
        "matched_route_count": matched_count,
        "memory_populations": report["memory_populations"],
        "primary_results": primary_results,
        "stochastic_aggregate": stochastic_aggregate,
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
