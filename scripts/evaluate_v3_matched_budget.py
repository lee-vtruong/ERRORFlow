"""Paper-oriented matched-budget routing ablation over cached B1/A3 outputs."""

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_memory import ResidualMemoryStore, record_from_dict
from errorflow.v3_routing import (
    best_memory_action,
    deterministic_random_score,
    estimate_action,
    select_budget,
    select_label_conditional_budget,
)
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap, percentile

try:
    from scripts.evaluate_v3_grouping_sensitivity import grouped_folds, near_duplicate_edges
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from evaluate_v3_grouping_sensitivity import grouped_folds, near_duplicate_edges


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def evidence_bucket(value):
    count = int(value or 0)
    return "0" if count == 0 else ("1" if count == 1 else "2+")


def per_class(expected, predicted):
    report = {}
    for label in sorted(set(expected) | set(predicted)):
        tp = sum(g == p == label for g, p in zip(expected, predicted))
        fp = sum(g != label and p == label for g, p in zip(expected, predicted))
        fn = sum(g == label and p != label for g, p in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": sum(g == label for g in expected),
        }
    return report


def evaluate_selection(ids, baseline, candidate, selected):
    gold = [baseline[key]["gold_label"] for key in ids]
    before = [baseline[key]["prediction"] for key in ids]
    after = [candidate[key]["prediction"] if key in selected else baseline[key]["prediction"] for key in ids]
    helpful = sum(b != g and a == g for g, b, a in zip(gold, before, after))
    harmful = sum(b == g and a != g for g, b, a in zip(gold, before, after))
    latencies = [
        float(baseline[key].get("latency_ms", 0.0)) +
        (float(candidate[key].get("latency_ms", 0.0)) if key in selected else 0.0)
        for key in ids
    ]
    generated = [
        int(baseline[key].get("generated_tokens", baseline[key].get("token_count", 0))) +
        (int(candidate[key].get("token_count", 0)) if key in selected else 0)
        for key in ids
    ]
    content_tokens = [
        (int(baseline[key].get("claim_tokens", 0)) + int(baseline[key].get("evidence_tokens", 0))) *
        (2 if key in selected else 1)
        for key in ids
    ]
    return {
        "n": len(ids),
        "routed": len(selected),
        "route_rate": len(selected) / len(ids),
        "accuracy": accuracy(after, gold),
        "macro_f1": macro_f1(after, gold),
        "per_class": per_class(gold, after),
        "helpful": helpful,
        "harmful": harmful,
        "net": helpful - harmful,
        "avg_forward_passes": 4.0 + 4.0 * len(selected) / len(ids),
        "mean_cached_component_latency_ms": sum(latencies) / len(latencies),
        "p95_cached_component_latency_ms": percentile(latencies, 0.95),
        "latency_measurement": "sum of cached B1 and routed A3 component latency; not fresh wall clock",
        "mean_generated_tokens": sum(generated) / len(generated),
        "mean_reprocessed_claim_evidence_whitespace_tokens": sum(content_tokens) / len(content_tokens),
        "model_tokenizer_input_tokens": None,
    }, after


def mean_interval(values):
    ordered = sorted(values)
    return {
        "mean": sum(values) / len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "empirical_95_interval": [percentile(ordered, 0.025), percentile(ordered, 0.975)],
    }


def holm_adjust(named_p_values):
    ordered = sorted(named_p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted, running = {}, 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (total - rank)))
        adjusted[name] = running
    return adjusted


def prepare_grouped_artifacts(baseline_rows, memory_rows, threshold, folds, seed):
    if threshold is None:
        assignments = {str(row["example_id"]): row["outer_fold"] for row in baseline_rows}
        return baseline_rows, [record_from_dict(row) for row in memory_rows], assignments, None
    edges = near_duplicate_edges(baseline_rows, threshold)
    assignments, components = grouped_folds(baseline_rows, edges, threshold, folds, seed)
    updated_baseline = []
    for source in baseline_rows:
        row = dict(source)
        row["outer_fold"] = assignments[str(row["example_id"])]
        updated_baseline.append(row)
    updated_memory = []
    for payload in memory_rows:
        record = record_from_dict(payload)
        provenance = dict(record.provenance)
        provenance["outer_fold"] = assignments[record.source_claim_id]
        updated_memory.append(replace(record, provenance=provenance))
    grouping = {
        "threshold": threshold,
        "candidate_edges": len(edges),
        "components": len(components),
        "multi_example_components": sum(len(component) > 1 for component in components),
        "largest_component": max(map(len, components)),
    }
    return updated_baseline, updated_memory, assignments, grouping


def build_scores(ids, baseline, records, action, top_k, min_support, cost_weight, random_seeds):
    store = ResidualMemoryStore(records)
    confidence, margin, similarity, memory, utility, static = {}, {}, {}, {}, {}, {}
    random = {seed: {} for seed in random_seeds}
    for key in ids:
        row = baseline[key]
        probabilities = row.get("label_probabilities", {})
        confidence[key] = 1.0 - max(probabilities.values()) if probabilities else -float(row.get("prediction_margin") or 0.0)
        margin[key] = -float(row.get("prediction_margin") or 0.0)
        for seed in random_seeds:
            random[seed][key] = deterministic_random_score(key, seed)
        query = {"claim": row["claim"], "retrieval_features": {
            "baseline_prediction": row["prediction"],
            "label_scores": row.get("label_scores", {}),
            "evidence_tokens": row.get("evidence_tokens", 0),
        }}
        neighbors = store.retrieve(
            query, top_k=top_k, action=action, exclude_claim_ids={key},
            exclude_folds={row["outer_fold"]},
        )
        similarity[key] = max((score for _, score in neighbors), default=float("-inf"))
        memory_best, _ = best_memory_action([action], neighbors, 0.0, min_support, False)
        utility_best, _ = best_memory_action([action], neighbors, cost_weight, min_support, True)
        memory[key] = memory_best.utility if memory_best else float("-inf")
        utility[key] = utility_best.utility if utility_best else float("-inf")
        bucket = evidence_bucket(row.get("evidence_count", 0))
        static_neighbors = [
            (record, 1.0) for record in records
            if record.correction_strategy == action
            and record.source_claim_id != key
            and record.provenance.get("outer_fold") != row["outer_fold"]
            and record.retrieval_features.get("baseline_prediction") == row["prediction"]
            and evidence_bucket(record.retrieval_features.get("evidence_count", 0)) == bucket
        ]
        static_estimate = estimate_action(action, static_neighbors, cost_weight)
        static[key] = static_estimate.utility if static_estimate.support >= min_support else float("-inf")
    return {
        "R-Conf": confidence,
        "R-Margin": margin,
        "R-Sim": similarity,
        "R-Static": static,
        "R-Memory": memory,
        "R6": utility,
    }, random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0, 0.05, 0.10, 0.1164, 0.15, 0.20, 0.35, 0.50, 1.0])
    parser.add_argument("--random-seeds", nargs="+", type=int, default=list(range(2026, 2046)))
    parser.add_argument("--group-threshold", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--action", default="conflicting_rescue_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    if any(not 0.0 <= value <= 1.0 for value in args.budgets):
        raise ValueError("budgets must be within [0, 1]")
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
    scores, random_scores = build_scores(
        ids, baseline, records, args.action, args.top_k, args.min_support,
        args.cost_weight, args.random_seeds,
    )
    labels = {key: baseline[key]["prediction"] for key in ids}
    positive_utility = {key for key in ids if scores["R6"][key] > args.utility_threshold}
    natural_budget = len(positive_utility) / len(ids)
    budgets = sorted(set(args.budgets + [natural_budget]))
    results, routes, prediction_cache = [], [], {}

    base_metrics, base_predictions = evaluate_selection(ids, baseline, candidate, set())
    results.append({"policy": "B1", "requested_budget": 0.0, **base_metrics})
    prediction_cache[("B1", 0.0)] = base_predictions
    all_metrics, all_predictions = evaluate_selection(ids, baseline, candidate, set(ids))
    results.append({"policy": "A3-All", "requested_budget": 1.0, **all_metrics})
    prediction_cache[("A3-All", 1.0)] = all_predictions

    deterministic = ["R-Conf", "R-Margin", "R-Label", "R-Sim", "R-Static", "R-Memory", "R6"]
    for budget in budgets:
        for policy in deterministic:
            if policy == "R-Label":
                selected = select_label_conditional_budget(ids, labels, scores["R-Margin"], budget)
            else:
                selected = select_budget(ids, scores[policy], budget)
            if policy == "R6":
                selected &= positive_utility
            metrics, predictions = evaluate_selection(ids, baseline, candidate, selected)
            results.append({"policy": policy, "requested_budget": budget, **metrics})
            prediction_cache[(policy, budget)] = predictions
            routes.append({"policy": policy, "requested_budget": budget, "selected_ids": sorted(selected)})
        seed_metrics = []
        for random_seed, random_score in random_scores.items():
            selected = select_budget(ids, random_score, budget)
            metrics, predictions = evaluate_selection(ids, baseline, candidate, selected)
            seed_metrics.append({"seed": random_seed, **metrics})
            if abs(budget - natural_budget) < 1e-12:
                prediction_cache[(f"R-Random:{random_seed}", budget)] = predictions
                routes.append({
                    "policy": f"R-Random:{random_seed}",
                    "requested_budget": budget,
                    "selected_ids": sorted(selected),
                })
        aggregate = {
            name: mean_interval([row[name] for row in seed_metrics])
            for name in ("accuracy", "macro_f1", "helpful", "harmful", "net", "route_rate", "avg_forward_passes")
        }
        results.append({
            "policy": "R-Random", "requested_budget": budget,
            "seeds": args.random_seeds, "aggregate": aggregate,
        })

    for row in results:
        if "accuracy" in row:
            row["delta_accuracy_vs_b1"] = row["accuracy"] - base_metrics["accuracy"]
            row["delta_macro_f1_vs_b1"] = row["macro_f1"] - base_metrics["macro_f1"]
        elif row["policy"] == "R-Random":
            row["aggregate"]["delta_accuracy_vs_b1"] = {
                "mean": row["aggregate"]["accuracy"]["mean"] - base_metrics["accuracy"]
            }
            row["aggregate"]["delta_macro_f1_vs_b1"] = {
                "mean": row["aggregate"]["macro_f1"]["mean"] - base_metrics["macro_f1"]
            }

    target_budget = natural_budget
    target_rows = {
        row["policy"]: row for row in results
        if row.get("requested_budget") == target_budget and row["policy"] != "R-Random"
    }
    non_memory = ["R-Conf", "R-Margin", "R-Label", "R-Sim", "R-Static"]
    strongest = max(non_memory, key=lambda name: (
        target_rows[name]["macro_f1"], target_rows[name]["accuracy"], name,
    ))
    gold = [baseline[key]["gold_label"] for key in ids]
    r6_predictions = prediction_cache[("R6", target_budget)]
    comparison_names = ["B1", "A3-All", "R-Conf", "R-Margin", "R-Label", "R-Sim", "R-Static", "R-Memory"]
    comparisons = {}
    raw_p_values = {}
    for name in comparison_names:
        key = (name, 0.0 if name == "B1" else (1.0 if name == "A3-All" else target_budget))
        comparator = prediction_cache[key]
        significance = {
            "mcnemar_exact": mcnemar_exact(gold, comparator, r6_predictions),
            "paired_bootstrap_r6_minus_comparator": paired_bootstrap(
                gold, comparator, r6_predictions, args.bootstrap_samples, args.seed,
            ),
        }
        comparisons[name] = significance
        raw_p_values[name] = significance["mcnemar_exact"]["exact_two_sided_p"]
    adjusted = holm_adjust(raw_p_values)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_mcnemar_p"] = value

    random_target = next(
        row for row in results
        if row["policy"] == "R-Random" and row["requested_budget"] == target_budget
    )
    random_comparison = {
        "seeds": args.random_seeds,
        "r6_accuracy_minus_random_mean": target_rows["R6"]["accuracy"] - random_target["aggregate"]["accuracy"]["mean"],
        "r6_macro_f1_minus_random_mean": target_rows["R6"]["macro_f1"] - random_target["aggregate"]["macro_f1"]["mean"],
        "r6_net_minus_random_mean": target_rows["R6"]["net"] - random_target["aggregate"]["net"]["mean"],
    }

    matched_predictions = []
    for (policy, budget), predictions in prediction_cache.items():
        if policy not in {"B1", "A3-All"} and abs(budget - target_budget) >= 1e-12:
            continue
        selected = set()
        if policy not in {"B1", "A3-All"}:
            route_row = next((row for row in routes if row["policy"] == policy and row["requested_budget"] == budget), None)
            selected = set(route_row["selected_ids"]) if route_row else {
                key for key, prediction in zip(ids, predictions) if prediction != baseline[key]["prediction"]
            }
        elif policy == "A3-All":
            selected = set(ids)
        for key, prediction in zip(ids, predictions):
            matched_predictions.append({
                "example_id": key, "policy": policy, "requested_budget": budget,
                "routed": key in selected, "prediction": prediction,
                "gold_label": baseline[key]["gold_label"],
            })

    output = {
        "evaluation": "near_duplicate_grouped_fold_excluded_matched_budget_cached_action_ablation",
        "configuration": {
            "action": args.action, "top_k": args.top_k, "min_support": args.min_support,
            "cost_weight": args.cost_weight, "utility_threshold": args.utility_threshold,
            "folds": args.folds, "seed": args.seed, "group_threshold": args.group_threshold,
            "budgets": budgets, "random_seeds": args.random_seeds,
            "latency_is_cached_component_sum": True,
        },
        "grouping": grouping,
        "natural_r6_budget": natural_budget,
        "strongest_matched_budget_non_memory_router": strongest,
        "results": results,
        "paired_comparisons_at_natural_r6_budget": comparisons,
        "random_comparison_at_natural_r6_budget": random_comparison,
        "multiple_comparison_correction": "Holm correction over exact McNemar tests listed above",
        "limitations": [
            "A3 was selected on an AVeriTeC train screen; this remains train-selected evidence.",
            "Cached component latency is not fresh end-to-end wall-clock latency.",
            "Tokenizer-level input token counts were not recorded; whitespace content-token processing is reported separately.",
        ],
    }
    atomic_write_json(args.output_dir / "summary.json", output)
    write_jsonl(args.output_dir / "routes.jsonl", routes)
    write_jsonl(args.output_dir / "natural_budget_predictions.jsonl", matched_predictions)
    print(json.dumps({
        "natural_r6_budget": natural_budget,
        "strongest_non_memory": strongest,
        "natural_budget_results": [
            row for row in results if row.get("requested_budget") == natural_budget
        ],
        "paired_comparisons": comparisons,
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
