"""Evaluate R6 similarity-component ablations from cached B1/A3 artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_memory import ResidualMemoryStore
from errorflow.v3_routing import best_memory_action
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap

try:
    from scripts.evaluate_v3_matched_budget import (
        evaluate_selection,
        indexed,
        prepare_grouped_artifacts,
    )
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from evaluate_v3_matched_budget import evaluate_selection, indexed, prepare_grouped_artifacts


VARIANTS = {
    "A0_full_r6": {"weights": (0.35, 0.50, 0.15), "selection_mode": "ranked"},
    "A1_without_claim": {"weights": (0.00, 0.50, 0.15), "selection_mode": "ranked"},
    "A2_without_label_score": {"weights": (0.35, 0.00, 0.15), "selection_mode": "ranked"},
    "A3_without_evidence_length": {"weights": (0.35, 0.50, 0.00), "selection_mode": "ranked"},
    "A4_label_score_only": {"weights": (0.00, 1.00, 0.00), "selection_mode": "ranked"},
    "A5_claim_only": {"weights": (1.00, 0.00, 0.00), "selection_mode": "ranked"},
    "A6_evidence_length_only": {"weights": (0.00, 0.00, 1.00), "selection_mode": "ranked"},
    "A7_uniform_weights": {"weights": (1.00, 1.00, 1.00), "selection_mode": "ranked"},
    "A8_random_neighbors": {"weights": (0.35, 0.50, 0.15), "selection_mode": "random"},
}


def utility_scores(ids, baseline, records, action, top_k, min_support, cost_weight,
                   weights, selection_mode="ranked", random_seed=2026):
    store = ResidualMemoryStore(records)
    scores, supports = {}, {}
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
            similarity_weights=weights,
            selection_mode=selection_mode,
            random_seed=random_seed,
        )
        best, _ = best_memory_action(
            [action], neighbors, cost_weight=cost_weight,
            min_support=min_support, utility_aware=True,
        )
        scores[key] = best.utility if best else float("-inf")
        supports[key] = best.support if best else 0
    return scores, supports


def top_count(ids, scores, count):
    finite = [key for key in ids if scores[key] != float("-inf")]
    if len(finite) < count:
        raise ValueError(f"Only {len(finite)} routable examples are available for matched count {count}")
    return set(sorted(finite, key=lambda key: (-scores[key], key))[:count])


def route_overlap(left, right):
    union = left | right
    return {
        "intersection": len(left & right),
        "union": len(union),
        "jaccard": len(left & right) / len(union) if union else 1.0,
    }


def summarize_replicates(rows):
    keys = (
        "routed", "route_rate", "accuracy", "macro_f1", "helpful", "harmful",
        "net", "avg_forward_passes", "delta_accuracy_vs_b1", "delta_macro_f1_vs_b1",
    )
    summary = {
        key: {
            "mean": sum(row[key] for row in rows) / len(rows),
            "min": min(row[key] for row in rows),
            "max": max(row[key] for row in rows),
        }
        for key in keys
    }
    overlaps = [row["route_overlap_with_a0_natural"]["jaccard"] for row in rows]
    summary["route_jaccard_with_a0_natural"] = {
        "mean": sum(overlaps) / len(overlaps), "min": min(overlaps), "max": max(overlaps),
    }
    return summary


def evaluate_variant(ids, baseline, candidate, selected, base_metrics):
    metrics, predictions = evaluate_selection(ids, baseline, candidate, selected)
    metrics["delta_accuracy_vs_b1"] = metrics["accuracy"] - base_metrics["accuracy"]
    metrics["delta_macro_f1_vs_b1"] = metrics["macro_f1"] - base_metrics["macro_f1"]
    return metrics, predictions


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
    parser.add_argument("--random-neighbor-seeds", nargs="+", type=int, default=list(range(2026, 2046)))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
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
    gold = [baseline[key]["gold_label"] for key in ids]

    full_scores, full_support = utility_scores(
        ids, baseline, records, args.action, args.top_k, args.min_support,
        args.cost_weight, **VARIANTS["A0_full_r6"], random_seed=args.seed,
    )
    full_natural = {key for key in ids if full_scores[key] > args.utility_threshold}
    matched_count = len(full_natural)
    results, route_rows, prediction_rows = {}, [], []

    def record_run(variant, mode, seed, selected, scores, supports, save_predictions=True):
        metrics, predictions = evaluate_variant(ids, baseline, candidate, selected, base_metrics)
        metrics["route_overlap_with_a0_natural"] = route_overlap(selected, full_natural)
        metrics["seed"] = seed
        results_key = f"{variant}:{mode}" + (f":seed={seed}" if seed is not None else "")
        results[results_key] = metrics
        route_rows.append({
            "variant": variant, "mode": mode, "seed": seed,
            "selected_ids": sorted(selected),
        })
        if save_predictions:
            for key, prediction in zip(ids, predictions):
                prediction_rows.append({
                    "example_id": key, "variant": variant, "mode": mode,
                    "seed": seed, "routed": key in selected,
                    "prediction": prediction, "gold_label": baseline[key]["gold_label"],
                    "estimated_utility": scores[key] if math.isfinite(scores[key]) else None,
                    "estimated_support": supports[key],
                })
        return metrics, predictions

    deterministic_variants = [name for name in VARIANTS if name != "A8_random_neighbors"]
    deterministic_predictions = {}
    for variant in deterministic_variants:
        if variant == "A0_full_r6":
            scores, supports = full_scores, full_support
        else:
            scores, supports = utility_scores(
                ids, baseline, records, args.action, args.top_k, args.min_support,
                args.cost_weight, **VARIANTS[variant], random_seed=args.seed,
            )
        natural = {key for key in ids if scores[key] > args.utility_threshold}
        matched = top_count(ids, scores, matched_count)
        _, natural_predictions = record_run(variant, "natural", None, natural, scores, supports)
        _, matched_predictions = record_run(variant, "matched_budget", None, matched, scores, supports)
        deterministic_predictions[(variant, "natural")] = natural_predictions
        deterministic_predictions[(variant, "matched_budget")] = matched_predictions

    random_results = {"natural": [], "matched_budget": []}
    for random_seed in args.random_neighbor_seeds:
        scores, supports = utility_scores(
            ids, baseline, records, args.action, args.top_k, args.min_support,
            args.cost_weight, **VARIANTS["A8_random_neighbors"], random_seed=random_seed,
        )
        for mode, selected in (
            ("natural", {key for key in ids if scores[key] > args.utility_threshold}),
            ("matched_budget", top_count(ids, scores, matched_count)),
        ):
            metrics, _ = record_run(
                "A8_random_neighbors", mode, random_seed, selected, scores, supports,
                save_predictions=False,
            )
            random_results[mode].append(metrics)

    a0_predictions = deterministic_predictions[("A0_full_r6", "matched_budget")]
    comparisons = {}
    for variant in deterministic_variants:
        if variant == "A0_full_r6":
            continue
        predictions = deterministic_predictions[(variant, "matched_budget")]
        comparisons[variant] = {
            "mcnemar_exact_a0_vs_variant": mcnemar_exact(gold, predictions, a0_predictions),
            "paired_bootstrap_a0_minus_variant": paired_bootstrap(
                gold, predictions, a0_predictions, args.bootstrap_samples, args.seed,
            ),
        }

    random_aggregate = {
        mode: summarize_replicates(rows) for mode, rows in random_results.items()
    }
    a0_matched = results["A0_full_r6:matched_budget"]
    a8_matched = random_aggregate["matched_budget"]
    output = {
        "evaluation": "grouped_fold_excluded_r6_similarity_component_ablation",
        "configuration": {
            "action": args.action,
            "top_k": args.top_k,
            "min_support": args.min_support,
            "cost_weight": args.cost_weight,
            "utility_threshold": args.utility_threshold,
            "group_threshold": args.group_threshold,
            "folds": args.folds,
            "seed": args.seed,
            "random_neighbor_seeds": args.random_neighbor_seeds,
            "matched_route_count": matched_count,
            "matched_route_rate": matched_count / len(ids),
            "similarity_weight_normalization": "weights normalized to sum to one",
            "random_neighbor_definition": "deterministic query-specific random compatible top-k; original full-similarity values retained for utility weighting",
        },
        "grouping": grouping,
        "baseline": base_metrics,
        "variants": VARIANTS,
        "results": results,
        "random_neighbor_aggregate": random_aggregate,
        "a8_matched_comparison_to_a0": {
            "a0_minus_a8_mean_accuracy": a0_matched["accuracy"] - a8_matched["accuracy"]["mean"],
            "a0_minus_a8_mean_macro_f1": a0_matched["macro_f1"] - a8_matched["macro_f1"]["mean"],
            "a0_minus_a8_mean_net": a0_matched["net"] - a8_matched["net"]["mean"],
        },
        "matched_budget_a0_comparisons": comparisons,
        "limitations": [
            "Development-fold ablation only; external target sets are not used for selection or tuning.",
            "Natural mode may change route count; matched-budget mode forces the A0 route count.",
            "Random-neighbor selection retains similarity weights after random selection to isolate ranking.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", output)
    write_jsonl(args.output_dir / "routes.jsonl", route_rows)
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)
    print(json.dumps({
        "matched_route_count": matched_count,
        "matched_route_rate": matched_count / len(ids),
        "deterministic_results": {
            key: value for key, value in results.items() if "seed=" not in key
        },
        "random_neighbor_aggregate": output["random_neighbor_aggregate"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
