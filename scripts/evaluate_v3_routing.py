"""Out-of-fold, matched-budget evaluation over cached V3 actions."""

import argparse
import json
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl
from errorflow.v3_memory import ResidualMemoryStore
from errorflow.v3_routing import (
    best_memory_action, deterministic_random_score, select_weighted_budget,
    select_label_conditional_weighted_budget,
)


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def weighted_metrics(expected, predicted, weights):
    labels = sorted(set(expected) | set(predicted))
    total = sum(weights)
    acc = sum(weight for gold, pred, weight in zip(expected, predicted, weights) if gold == pred) / total
    f1s = []
    for label in labels:
        tp = sum(w for g, p, w in zip(expected, predicted, weights) if g == p == label)
        fp = sum(w for g, p, w in zip(expected, predicted, weights) if g != label and p == label)
        fn = sum(w for g, p, w in zip(expected, predicted, weights) if g == label and p != label)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return acc, sum(f1s) / len(f1s)


def group_name(row):
    return f"control:{row['gold_label']}" if row["correct"] else f"error:{row['gold_label']}->{row['prediction']}"


def evaluate(ids, baseline, final, routed, actions, query_weights):
    gold = [baseline[key]["gold_label"] for key in ids]
    before = [baseline[key]["prediction"] for key in ids]
    after = [final[key] for key in ids]
    weights = [query_weights[key] for key in ids]
    base_accuracy, base_f1 = weighted_metrics(gold, before, weights)
    final_accuracy, final_f1 = weighted_metrics(gold, after, weights)
    helpful = sum(w for g, b, a, w in zip(gold, before, after, weights) if b != g and a == g)
    harmful = sum(w for g, b, a, w in zip(gold, before, after, weights) if b == g and a != g)
    total_weight = sum(weights)
    routed_weight = sum(query_weights[key] for key in routed)
    return {
        "n_screen": len(ids), "population_weight": total_weight,
        "accuracy": final_accuracy, "macro_f1": final_f1,
        "delta_accuracy": final_accuracy - base_accuracy, "delta_macro_f1": final_f1 - base_f1,
        "weighted_helpful": helpful, "weighted_harmful": harmful, "weighted_net": helpful - harmful,
        "weighted_route_rate": routed_weight / total_weight if total_weight else 0.0,
        "weighted_action_routes": {
            action: sum(query_weights[key] for key in routed if actions.get(key) == action)
            for action in sorted(set(actions.values()))
        },
        "avg_forward_passes": 4.0 + 4.0 * routed_weight / total_weight if total_weight else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--population-records", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--candidate", nargs="+", required=True, help="ACTION=predictions.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--memory-size", type=int, choices=[0, 25, 50, 100, 200])
    parser.add_argument("--default-action", default="conflicting_rescue_v1")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    baseline = indexed(read_jsonl(args.baseline_records), "baseline_records")
    population = indexed(read_jsonl(args.population_records), "population_records")
    candidates = {}
    for specification in args.candidate:
        action, separator, filename = specification.partition("=")
        if not separator:
            raise ValueError(f"Expected ACTION=PATH, got {specification}")
        candidates[action] = indexed(read_jsonl(filename), action)
    ids = sorted(baseline)
    if args.default_action not in candidates:
        raise ValueError("default action must be one of the candidate actions")
    if any(set(rows) != set(ids) for rows in candidates.values()):
        raise ValueError("Every candidate must exactly match baseline IDs")
    store = ResidualMemoryStore.load(args.memory)
    memory_actions, memory_scores, utility_actions, utility_scores = {}, {}, {}, {}
    confidence_scores = {key: -float(baseline[key].get("prediction_margin") or 0.0) for key in ids}
    random_scores = {key: deterministic_random_score(key, args.seed) for key in ids}
    labels = {key: baseline[key]["prediction"] for key in ids}
    population_counts = {}
    screen_counts = {}
    for row in population.values():
        population_counts[group_name(row)] = population_counts.get(group_name(row), 0) + 1
    for row in baseline.values():
        screen_counts[group_name(row)] = screen_counts.get(group_name(row), 0) + 1
    missing_groups = set(population_counts) - set(screen_counts)
    if missing_groups:
        raise ValueError(f"Screen does not represent population groups: {sorted(missing_groups)}")
    query_weights = {key: population_counts[group_name(row)] / screen_counts[group_name(row)] for key, row in baseline.items()}
    for key in ids:
        row = baseline[key]
        query = {"claim": row["claim"], "retrieval_features": {
            "baseline_prediction": row["prediction"], "label_scores": row.get("label_scores", {}),
            "evidence_tokens": row.get("evidence_tokens", 0),
        }}
        neighbors = store.retrieve(
            query, top_k=args.top_k * len(candidates), exclude_claim_ids={key},
            exclude_folds={row.get("outer_fold")}, max_records=args.memory_size,
        ) if args.memory_size != 0 else []
        memory_best, _ = best_memory_action(candidates, neighbors, 0.0, args.min_support, False)
        utility_best, _ = best_memory_action(candidates, neighbors, args.cost_weight, args.min_support, True)
        if memory_best:
            memory_actions[key], memory_scores[key] = memory_best.action, memory_best.utility
        if utility_best:
            utility_actions[key], utility_scores[key] = utility_best.action, utility_best.utility

    results = []
    baseline_final = {key: baseline[key]["prediction"] for key in ids}
    results.append({"policy": "R0_never", "budget": 0.0, **evaluate(ids, baseline, baseline_final, set(), {}, query_weights)})
    for budget in sorted(set(args.budgets)):
        policies = {
            "R2_random": (select_weighted_budget(ids, random_scores, query_weights, budget), {key: args.default_action for key in ids}),
            "R3_confidence": (select_weighted_budget(ids, confidence_scores, query_weights, budget), {key: args.default_action for key in ids}),
            "R4_label_conditional": (select_label_conditional_weighted_budget(ids, labels, confidence_scores, query_weights, budget), {key: args.default_action for key in ids}),
            "R5_memory": (select_weighted_budget(ids, memory_scores, query_weights, budget) & set(memory_actions), memory_actions),
            "R6_utility": (select_weighted_budget(ids, utility_scores, query_weights, budget) & {key for key, score in utility_scores.items() if score > 0}, utility_actions),
        }
        for policy, (routed, action_map) in policies.items():
            final = {key: (candidates[action_map[key]][key]["prediction"] if key in routed else baseline[key]["prediction"]) for key in ids}
            results.append({"policy": policy, "budget": budget, **evaluate(ids, baseline, final, routed, action_map, query_weights)})
    all_routed = set(ids)
    always_final = {key: candidates[args.default_action][key]["prediction"] for key in ids}
    results.append({"policy": "R1_always", "budget": 1.0,
                    **evaluate(ids, baseline, always_final, all_routed, {key: args.default_action for key in ids}, query_weights)})
    output = {
        "evaluation": "grouped_out_of_fold_cached_action_routing",
        "seed": args.seed, "cost_weight": args.cost_weight, "top_k": args.top_k,
        "min_support": args.min_support, "memory_size": args.memory_size,
        "default_action": args.default_action, "population_examples": len(population), "screen_examples": len(baseline),
        "population_group_counts": population_counts, "screen_group_counts": screen_counts, "results": results,
        "limitations": ["Outer-fold exclusion is enforced; nested inner-fold hyperparameter selection is not implemented in this command."],
    }
    atomic_write_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
