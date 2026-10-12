"""Bootstrap and negative-control checks for a frozen V2 memory router."""

import argparse
import json
import random
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
    accuracy = sum(weight for gold, pred, weight in zip(expected, predicted, weights)
                   if gold == pred) / total
    f1s = []
    for label in labels:
        tp = sum(w for g, p, w in zip(expected, predicted, weights) if g == p == label)
        fp = sum(w for g, p, w in zip(expected, predicted, weights) if g != label and p == label)
        fn = sum(w for g, p, w in zip(expected, predicted, weights) if g == label and p != label)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return accuracy, sum(f1s) / len(f1s) if f1s else 0.0


def percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def evaluate_rows(ids, gold, before, after, weights):
    expected = [gold[key]["label"] for key in ids]
    baseline_labels = [before[key]["prediction"] for key in ids]
    candidate_labels = [after[key] for key in ids]
    baseline_accuracy, baseline_macro_f1 = weighted_metrics(expected, baseline_labels, weights)
    candidate_accuracy, candidate_macro_f1 = weighted_metrics(expected, candidate_labels, weights)
    helpful = sum(weight for key, weight in zip(ids, weights)
                  if before[key]["prediction"] != gold[key]["label"] and after[key] == gold[key]["label"])
    harmful = sum(weight for key, weight in zip(ids, weights)
                  if before[key]["prediction"] == gold[key]["label"] and after[key] != gold[key]["label"])
    return {
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "delta_accuracy": candidate_accuracy - baseline_accuracy,
        "baseline_macro_f1": baseline_macro_f1,
        "candidate_macro_f1": candidate_macro_f1,
        "delta_macro_f1": candidate_macro_f1 - baseline_macro_f1,
        "weighted_helpful": helpful,
        "weighted_harmful": harmful,
        "weighted_net": helpful - harmful,
    }


def stratified_bootstrap(ids, gold, before, after, query_weights, samples, seed):
    groups = defaultdict(list)
    for key in ids:
        groups[group_name(gold[key]["label"], before[key]["prediction"])].append(key)
    rng = random.Random(seed)
    accuracy_deltas, macro_f1_deltas = [], []
    for _ in range(samples):
        sampled = []
        for name in sorted(groups):
            members = groups[name]
            sampled.extend(rng.choice(members) for _ in members)
        metrics = evaluate_rows(sampled, gold, before, after, [query_weights[key] for key in sampled])
        accuracy_deltas.append(metrics["delta_accuracy"])
        macro_f1_deltas.append(metrics["delta_macro_f1"])
    return {
        "samples": samples,
        "seed": seed,
        "stratification": "offline_gold_to_baseline_transition",
        "accuracy_delta_ci95": [percentile(accuracy_deltas, 0.025), percentile(accuracy_deltas, 0.975)],
        "macro_f1_delta_ci95": [percentile(macro_f1_deltas, 0.025), percentile(macro_f1_deltas, 0.975)],
        "p_delta_accuracy_gt_0": sum(value > 0 for value in accuracy_deltas) / samples,
        "p_delta_macro_f1_gt_0": sum(value > 0 for value in macro_f1_deltas) / samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-gold", required=True, type=Path)
    parser.add_argument("--screen-gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--candidate", nargs="+", required=True, help="ACTION=predictions.jsonl")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--random-trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    population = indexed(read_jsonl(args.population_gold), "population_gold")
    screen = indexed(read_jsonl(args.screen_gold), "screen_gold")
    baseline = indexed(read_jsonl(args.baseline), "baseline")
    candidates = {}
    for specification in args.candidate:
        action, separator, filename = specification.partition("=")
        if not separator:
            raise ValueError(f"Expected ACTION=PATH, got {specification}")
        candidates[action] = indexed(read_jsonl(filename), action)
    expected_actions = set(config["actions"])
    if set(candidates) != expected_actions:
        raise ValueError(f"Candidate actions {sorted(candidates)} do not match frozen config {sorted(expected_actions)}")
    store = ErrorMemoryStore.load(args.memory)

    population_counts = Counter(group_name(row["label"], baseline[key]["prediction"])
                                for key, row in population.items())
    screen_counts = Counter(group_name(row["label"], baseline[key]["prediction"])
                            for key, row in screen.items())
    query_weights = {
        key: population_counts[group_name(row["label"], baseline[key]["prediction"])]
        / screen_counts[group_name(row["label"], baseline[key]["prediction"])]
        for key, row in screen.items()
    }
    ids = sorted(screen)
    weights = [query_weights[key] for key in ids]
    route_actions, final = {}, {}
    for key in ids:
        row, before = screen[key], baseline[key]
        neighbors = []
        for action in config["actions"]:
            neighbors.extend(store.retrieve(
                row["claim"], dataset=row.get("dataset"), top_k=config["top_k_per_action"],
                allowed_splits=set(config["memory_splits"]), baseline_prediction=before["prediction"],
                label_scores=before.get("metadata", {}).get("label_scores"),
                exclude_claim_ids={key}, action=action,
            ))
        confidence = before.get("confidence")
        uncertainty = 1.0 - float(confidence) if confidence is not None else 1.0
        decision = select_action(
            neighbors, uncertainty=uncertainty, cost_weight=config["cost_weight"],
            min_support=config["min_support"], min_similarity=config["min_similarity"],
            min_utility=config["min_utility"],
        )
        route_actions[key] = decision.action if decision.route else "none"
        final[key] = candidates[decision.action][key]["prediction"] if decision.route else before["prediction"]

    frozen = evaluate_rows(ids, screen, baseline, final, weights)
    routed_weight = sum(query_weights[key] for key in ids if route_actions[key] != "none")
    frozen.update({
        "weighted_route_rate": routed_weight / sum(weights),
        "weighted_action_routes": {
            action: sum(query_weights[key] for key in ids if route_actions[key] == action)
            for action in config["actions"]
        },
    })

    ablations = {"no_memory": evaluate_rows(
        ids, screen, baseline, {key: baseline[key]["prediction"] for key in ids}, weights
    )}
    for action in config["actions"]:
        ablations[f"always_{action}"] = evaluate_rows(
            ids, screen, baseline, {key: candidates[action][key]["prediction"] for key in ids}, weights
        )

    routed_count = sum(action != "none" for action in route_actions.values())
    random_deltas, random_route_rates = [], []
    rng = random.Random(args.seed)
    for _ in range(args.random_trials):
        chosen = {key for key in ids if rng.random() < frozen["weighted_route_rate"]}
        random_final = {
            key: candidates[rng.choice(config["actions"])][key]["prediction"]
            if key in chosen else baseline[key]["prediction"] for key in ids
        }
        random_deltas.append(evaluate_rows(ids, screen, baseline, random_final, weights)["delta_accuracy"])
        random_route_rates.append(sum(query_weights[key] for key in chosen) / sum(weights))
    ablations["random_route_same_expected_budget"] = {
        "trials": args.random_trials,
        "frozen_routed_screen_examples": routed_count,
        "target_weighted_route_rate": frozen["weighted_route_rate"],
        "mean_weighted_route_rate": sum(random_route_rates) / len(random_route_rates),
        "mean_delta_accuracy": sum(random_deltas) / len(random_deltas),
        "delta_accuracy_ci95": [percentile(random_deltas, 0.025), percentile(random_deltas, 0.975)],
        "p_random_delta_ge_frozen": sum(value >= frozen["delta_accuracy"] for value in random_deltas) / len(random_deltas),
    }

    result = {
        "evaluation": "frozen_router_train_only_robustness",
        "screen_examples": len(screen),
        "population_examples": len(population),
        "config": config,
        "frozen_router": frozen,
        "bootstrap": stratified_bootstrap(
            ids, screen, baseline, final, query_weights, args.bootstrap_samples, args.seed
        ),
        "ablations": ablations,
        "holdout_labels_read": False,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
