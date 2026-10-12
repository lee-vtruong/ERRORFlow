"""Population-weighted leave-one-example-out evaluation of the memory router."""

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from errorflow.error_memory import ErrorMemoryStore
from errorflow.io import atomic_write_json, read_jsonl
from errorflow.routing_v2 import select_action


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def group_name(expected, predicted):
    return f"control:{expected}" if expected == predicted else f"error:{expected}->{predicted}"


def weighted_metrics(expected, predicted, weights):
    labels = sorted(set(expected) | set(predicted))
    total = sum(weights)
    accuracy = sum(weight for gold, pred, weight in zip(expected, predicted, weights) if gold == pred) / total
    f1s = []
    for label in labels:
        tp = sum(w for g, p, w in zip(expected, predicted, weights) if g == p == label)
        fp = sum(w for g, p, w in zip(expected, predicted, weights) if g != label and p == label)
        fn = sum(w for g, p, w in zip(expected, predicted, weights) if g == label and p != label)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return accuracy, sum(f1s) / len(f1s) if f1s else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-gold", required=True, type=Path)
    parser.add_argument("--screen-gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--candidate", nargs="+", required=True, help="ACTION=predictions.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    population = indexed(read_jsonl(args.population_gold), "population_gold")
    screen = indexed(read_jsonl(args.screen_gold), "screen_gold")
    baseline = indexed(read_jsonl(args.baseline), "baseline")
    candidates = {}
    for specification in args.candidate:
        action, separator, filename = specification.partition("=")
        if not separator:
            raise ValueError(f"Expected ACTION=PATH, got {specification}")
        candidates[action] = indexed(read_jsonl(filename), action)
    store = ErrorMemoryStore.load(args.memory)

    population_counts = Counter(group_name(row["label"], baseline[key]["prediction"]) for key, row in population.items())
    screen_counts = Counter(group_name(row["label"], baseline[key]["prediction"]) for key, row in screen.items())
    query_weights = {key: population_counts[group_name(row["label"], baseline[key]["prediction"])]
                     / screen_counts[group_name(row["label"], baseline[key]["prediction"])] for key, row in screen.items()}

    top_ks = (3, 5, 10, 20)
    max_top_k = max(top_ks)
    neighbor_cache = {}
    for key, row in screen.items():
        before = baseline[key]
        for action in candidates:
            neighbor_cache[(key, action)] = store.retrieve(
                row["claim"], dataset=row.get("dataset"), top_k=max_top_k,
                allowed_splits={"train"}, baseline_prediction=before["prediction"],
                label_scores=before.get("metadata", {}).get("label_scores"),
                exclude_claim_ids={key}, action=action,
            )

    configurations = []
    grid = itertools.product(top_ks, (2, 3, 5), (0.0, 0.05, 0.1), (0.0, 0.05, 0.1), ("full", "inverse_confidence"))
    ids = sorted(screen)
    expected = [screen[key]["label"] for key in ids]
    baseline_labels = [baseline[key]["prediction"] for key in ids]
    weights = [query_weights[key] for key in ids]
    baseline_accuracy, baseline_macro_f1 = weighted_metrics(expected, baseline_labels, weights)
    for top_k, min_support, min_utility, cost_weight, uncertainty_mode in grid:
        final, routes, helpful, harmful = [], Counter(), 0.0, 0.0
        for key, gold_label, before_label, weight in zip(ids, expected, baseline_labels, weights):
            neighbors = []
            for action in candidates:
                neighbors.extend(neighbor_cache[(key, action)][:top_k])
            confidence = baseline[key].get("confidence")
            uncertainty = 1.0 if uncertainty_mode == "full" or confidence is None else 1.0 - float(confidence)
            decision = select_action(
                neighbors, uncertainty=uncertainty, cost_weight=cost_weight,
                min_support=min_support, min_similarity=0.0, min_utility=min_utility,
            )
            after = candidates[decision.action][key]["prediction"] if decision.route else before_label
            final.append(after)
            if decision.route:
                routes[decision.action] += weight
            helpful += weight * (before_label != gold_label and after == gold_label)
            harmful += weight * (before_label == gold_label and after != gold_label)
        candidate_accuracy, candidate_macro_f1 = weighted_metrics(expected, final, weights)
        total_weight = sum(weights)
        configurations.append({
            "top_k_per_action": top_k, "min_support": min_support,
            "min_utility": min_utility, "cost_weight": cost_weight,
            "uncertainty_mode": uncertainty_mode,
            "baseline_accuracy": baseline_accuracy, "candidate_accuracy": candidate_accuracy,
            "delta_accuracy": candidate_accuracy - baseline_accuracy,
            "baseline_macro_f1": baseline_macro_f1, "candidate_macro_f1": candidate_macro_f1,
            "delta_macro_f1": candidate_macro_f1 - baseline_macro_f1,
            "weighted_helpful": helpful, "weighted_harmful": harmful,
            "weighted_net": helpful - harmful,
            "weighted_route_rate": sum(routes.values()) / total_weight,
            "weighted_action_routes": dict(routes),
        })
    configurations.sort(key=lambda row: (-row["candidate_accuracy"], -row["candidate_macro_f1"], row["weighted_route_rate"]))
    result = {
        "evaluation": "leave_one_example_out_population_weighted_train_screen",
        "screen_examples": len(screen), "population_examples": len(population),
        "population_group_counts": dict(sorted(population_counts.items())),
        "screen_group_counts": dict(sorted(screen_counts.items())),
        "best": configurations[0], "top10": configurations[:10],
        "n_configurations": len(configurations),
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
