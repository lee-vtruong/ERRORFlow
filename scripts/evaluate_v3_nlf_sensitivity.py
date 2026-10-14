"""Evaluate N-E sensitivity/robustness for the final R6-NLF router.

All experiments replay cached predictions.  Memory-size units are total source
records before target-fold exclusion; random subsets are repeated over seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl, write_jsonl

try:
    from scripts.evaluate_v3_matched_budget import (
        evaluate_selection, indexed, prepare_grouped_artifacts,
    )
    from scripts.evaluate_v3_substitution_ablation import Engine, top_count, variant
except ModuleNotFoundError:
    from evaluate_v3_matched_budget import evaluate_selection, indexed, prepare_grouped_artifacts
    from evaluate_v3_substitution_ablation import Engine, top_count, variant


REFERENCE = variant(compatibility="none")


def stable_sample(records, count, seed):
    if count > len(records):
        raise ValueError(f"memory size {count} exceeds {len(records)} records")
    return sorted(records, key=lambda record: hashlib.sha256(
        f"NE4:{seed}:{record.memory_id}".encode("utf-8")
    ).hexdigest())[:count]


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def finite_selection(ids, scores, threshold=0.0):
    return {key for key in ids if math.isfinite(scores[key]) and scores[key] > threshold}


def utility_bin(value):
    boundaries = (-0.10, -0.05, -0.025, -0.01, 0.0, 0.01, 0.025, 0.05, 0.10)
    labels = (
        "u<-0.10", "-0.10<=u<-0.05", "-0.05<=u<-0.025",
        "-0.025<=u<-0.01", "-0.01<=u<0", "0<=u<0.01",
        "0.01<=u<0.025", "0.025<=u<0.05", "0.05<=u<0.10", "u>=0.10",
    )
    for index, upper in enumerate(boundaries):
        if value < upper:
            return labels[index]
    return labels[-1]


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
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--min-supports", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument(
        "--budgets", nargs="+", type=float,
        default=[0.05, 0.10, 0.1222, 0.15, 0.20, 0.30, 0.40],
    )
    parser.add_argument(
        "--memory-sizes", nargs="+", type=int,
        default=[25, 50, 100, 200, 500, 1000],
    )
    parser.add_argument("--random-seeds", nargs="+", type=int, default=list(range(2026, 2046)))
    parser.add_argument("--group-thresholds", nargs="+", type=float, default=[0.85, 0.90, 0.95])
    parser.add_argument("--group-threshold", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if any(value <= 0 for value in args.top_ks + args.min_supports + args.memory_sizes):
        raise ValueError("top-k, min-support, and memory-size values must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.budgets):
        raise ValueError("budgets must lie in [0,1]")
    if args.seed not in args.random_seeds:
        raise ValueError("--seed must appear in --random-seeds")

    raw_baseline = list(read_jsonl(args.baseline_records))
    raw_memory = list(read_jsonl(args.memory))
    baseline_rows, records, assignments, grouping = prepare_grouped_artifacts(
        raw_baseline, raw_memory, args.group_threshold, args.folds, args.seed,
    )
    baseline = indexed(baseline_rows, "baseline")
    candidate = indexed(read_jsonl(args.candidate), "candidate")
    examples = indexed(read_jsonl(args.examples), "examples")
    if set(baseline) != set(candidate) or set(baseline) != set(examples):
        raise ValueError("Baseline, candidate, and examples IDs must match exactly")
    ids = sorted(baseline)
    engine = Engine(ids, baseline, records, examples, args.action, args.cost_weight)
    base_metrics, _ = evaluate_selection(ids, baseline, candidate, set())
    sensitivity_rows, prediction_rows, route_rows = [], [], []

    def evaluate_point(experiment, setting, scores, supports, *, seed=None, save=True):
        selected = finite_selection(ids, scores, args.utility_threshold)
        metrics, predictions = evaluate_selection(ids, baseline, candidate, selected)
        row = {
            "experiment": experiment, "setting": setting, "seed": seed,
            "memory_records_before_fold_exclusion": len(engine.records),
            **metrics,
        }
        sensitivity_rows.append(row)
        route_rows.append({
            "experiment": experiment, "setting": setting, "seed": seed,
            "selected_ids": sorted(selected),
        })
        if save:
            for key, prediction in zip(ids, predictions):
                prediction_rows.append({
                    "example_id": key, "outer_fold": assignments[key],
                    "experiment": experiment, "setting": setting, "seed": seed,
                    "routed": key in selected, "prediction": prediction,
                    "gold_label": baseline[key]["gold_label"],
                    "estimated_utility": scores[key] if math.isfinite(scores[key]) else None,
                    "estimated_support": supports[key],
                })
        return row, selected

    # NE1: top-k.  k=1 automatically receives min_support=1.
    ne1_cache = {}
    for top_k in args.top_ks:
        config = variant(
            compatibility="none", top_k=top_k, min_support=min(2, top_k),
        )
        print(f"NE1 top_k={top_k}", flush=True)
        scores, supports = engine.score_variant(config)
        ne1_cache[top_k] = (scores, supports)
        evaluate_point("NE1_top_k", str(top_k), scores, supports)

    # Canonical reference is shared by later analyses.
    if 5 in ne1_cache:
        reference_scores, reference_supports = ne1_cache[5]
    else:
        reference_scores, reference_supports = engine.score_variant(REFERENCE)

    # NE2: minimum support at fixed k=5.
    for support in args.min_supports:
        if support > 5:
            raise ValueError("NE2 min support cannot exceed fixed top_k=5")
        config = variant(compatibility="none", top_k=5, min_support=support)
        print(f"NE2 min_support={support}", flush=True)
        scores, supports = engine.score_variant(config)
        evaluate_point("NE2_min_support", str(support), scores, supports)

    # NE3: exact matched routing budgets, ranked by canonical utility.
    for budget in args.budgets:
        count = min(len(ids), round(budget * len(ids)))
        selected = top_count(ids, reference_scores, count)
        metrics, predictions = evaluate_selection(ids, baseline, candidate, selected)
        sensitivity_rows.append({
            "experiment": "NE3_routing_budget", "setting": budget, "seed": None,
            "requested_count": count, **metrics,
        })
        route_rows.append({
            "experiment": "NE3_routing_budget", "setting": budget,
            "seed": None, "selected_ids": sorted(selected),
        })
        for key, prediction in zip(ids, predictions):
            prediction_rows.append({
                "example_id": key, "outer_fold": assignments[key],
                "experiment": "NE3_routing_budget", "setting": budget,
                "seed": None, "routed": key in selected, "prediction": prediction,
                "gold_label": baseline[key]["gold_label"],
                "estimated_utility": reference_scores[key],
                "estimated_support": reference_supports[key],
            })

    # NE4: total memory records before fold exclusion, repeated over seeds.
    full_records = list(engine.records)
    memory_seed_rows = []
    for size in args.memory_sizes:
        for random_seed in args.random_seeds:
            engine.records = stable_sample(full_records, size, random_seed)
            print(f"NE4 memory_size={size} seed={random_seed}", flush=True)
            scores, supports = engine.score_variant(REFERENCE)
            row, _ = evaluate_point(
                "NE4_memory_size", str(size), scores, supports,
                seed=random_seed, save=True,
            )
            row["mean_eligible_records_after_fold_exclusion"] = statistics.mean(
                sum(record.provenance.get("outer_fold") != baseline[key]["outer_fold"]
                    for record in engine.records)
                for key in ids
            )
            memory_seed_rows.append(row)
    engine.records = full_records
    full_row, _ = evaluate_point(
        "NE4_memory_size", "full", reference_scores, reference_supports,
    )
    full_row["mean_eligible_records_after_fold_exclusion"] = statistics.mean(
        sum(record.provenance.get("outer_fold") != baseline[key]["outer_fold"]
            for record in full_records)
        for key in ids
    )
    memory_aggregate = {}
    for size in args.memory_sizes:
        rows = [row for row in memory_seed_rows if row["setting"] == str(size)]
        memory_aggregate[str(size)] = {
            key: summarize([float(row[key]) for row in rows])
            for key in ("route_rate", "accuracy", "macro_f1", "helpful", "harmful", "net")
        }
    memory_aggregate["full"] = {key: full_row[key] for key in (
        "route_rate", "accuracy", "macro_f1", "helpful", "harmful", "net",
    )}

    # NE5: recompute grouping/folds at each threshold from the untouched inputs.
    group_results = []
    for threshold in args.group_thresholds:
        grouped_rows, grouped_records, grouped_assignments, grouped_info = prepare_grouped_artifacts(
            raw_baseline, raw_memory, threshold, args.folds, args.seed,
        )
        grouped_baseline = indexed(grouped_rows, "grouped baseline")
        grouped_engine = Engine(
            ids, grouped_baseline, grouped_records, examples, args.action, args.cost_weight,
        )
        print(f"NE5 group_threshold={threshold}", flush=True)
        scores, supports = grouped_engine.score_variant(REFERENCE)
        selected = finite_selection(ids, scores, args.utility_threshold)
        metrics, predictions = evaluate_selection(ids, grouped_baseline, candidate, selected)
        group_results.append({"threshold": threshold, "grouping": grouped_info, **metrics})
        route_rows.append({
            "experiment": "NE5_group_threshold", "setting": threshold,
            "seed": None, "selected_ids": sorted(selected),
        })
        for key, prediction in zip(ids, predictions):
            prediction_rows.append({
                "example_id": key, "outer_fold": grouped_assignments[key],
                "experiment": "NE5_group_threshold", "setting": threshold,
                "seed": None, "routed": key in selected, "prediction": prediction,
                "gold_label": grouped_baseline[key]["gold_label"],
                "estimated_utility": scores[key] if math.isfinite(scores[key]) else None,
                "estimated_support": supports[key],
            })

    # NE7: empirical correction outcomes as a function of distance to U=0.
    uncertainty_groups = defaultdict(list)
    for key in ids:
        score = reference_scores[key]
        if math.isfinite(score):
            uncertainty_groups[utility_bin(score)].append(key)
    uncertainty = []
    for bucket, keys in sorted(uncertainty_groups.items()):
        helpful = harmful = neutral = 0
        for key in keys:
            before = baseline[key]["prediction"] == baseline[key]["gold_label"]
            after = candidate[key]["prediction"] == baseline[key]["gold_label"]
            helpful += int(not before and after)
            harmful += int(before and not after)
            neutral += int(before == after)
        uncertainty.append({
            "utility_margin_bin": bucket, "n": len(keys),
            "mean_utility": statistics.mean(reference_scores[key] for key in keys),
            "routed_at_zero": sum(reference_scores[key] > 0 for key in keys),
            "counterfactual_helpful": helpful,
            "counterfactual_harmful": harmful,
            "counterfactual_neutral": neutral,
            "empirical_net_rate": (helpful - harmful) / len(keys),
        })

    summary = {
        "evaluation": "grouped_fold_excluded_nlf_sensitivity_robustness",
        "configuration": {
            "action": args.action, "compatibility": "none",
            "reference_top_k": 5, "reference_min_support": 2,
            "cost_weight": args.cost_weight, "utility_threshold": args.utility_threshold,
            "group_threshold": args.group_threshold, "folds": args.folds,
            "seed": args.seed, "random_seeds": args.random_seeds,
            "memory_size_unit": "total source records before target-fold exclusion",
        },
        "grouping": grouping, "baseline": base_metrics,
        "results": sensitivity_rows,
        "memory_size_aggregate": memory_aggregate,
        "group_threshold_results": group_results,
        "utility_uncertainty": uncertainty,
        "protocol_notes": [
            "NE1 uses min_support=min(2, top_k), so top_k=1 remains routable.",
            "NE3 selects the exact requested count by canonical utility rank, including non-positive scores when required by the budget.",
            "NE4 samples a total source-memory size before fold exclusion and reports variability over every declared seed.",
            "NE5 changes only duplicate-group threshold and does not tune on external evaluation sets.",
            "NE6 is intentionally evaluated by evaluate_v3_external_memory_sensitivity.py on locked external targets, not on this source population.",
            "NE7 uses cached gold labels only for post-hoc empirical outcome analysis, never for routing.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(args.output_dir / "routes.jsonl", route_rows)
    write_jsonl(args.output_dir / "sensitivity.jsonl", sensitivity_rows)
    print(json.dumps({
        "sensitivity_rows": len(sensitivity_rows),
        "prediction_rows": len(prediction_rows),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
