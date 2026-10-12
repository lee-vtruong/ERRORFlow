"""Summarize helpful and harmful outcomes from a bounded V3 action screen."""

import argparse
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def group_name(row):
    if row["correct"]:
        return f"control:{row['gold_label']}"
    return f"error:{row['gold_label']}->{row['prediction']}"


def weighted_metrics(gold, predicted, weights):
    labels = sorted(set(gold) | set(predicted))
    total = sum(weights)
    accuracy = sum(w for g, p, w in zip(gold, predicted, weights) if g == p) / total
    f1s = []
    for label in labels:
        tp = sum(w for g, p, w in zip(gold, predicted, weights) if g == p == label)
        fp = sum(w for g, p, w in zip(gold, predicted, weights) if g != label and p == label)
        fn = sum(w for g, p, w in zip(gold, predicted, weights) if g == label and p != label)
        denominator = 2 * tp + fp + fn
        f1s.append(2 * tp / denominator if denominator else 0.0)
    return accuracy, sum(f1s) / len(f1s)


def summarize_action(baseline, candidate, population_counts, screen_counts):
    ids = sorted(baseline)
    if set(candidate) != set(ids):
        missing = set(ids) - set(candidate)
        extra = set(candidate) - set(ids)
        raise ValueError(f"Candidate IDs differ: missing={len(missing)} extra={len(extra)}")
    weights = {key: population_counts[group_name(row)] / screen_counts[group_name(row)]
               for key, row in baseline.items()}
    gold = [baseline[key]["gold_label"] for key in ids]
    before = [baseline[key]["prediction"] for key in ids]
    after = [candidate[key]["prediction"] for key in ids]
    row_weights = [weights[key] for key in ids]
    base_acc, base_f1 = weighted_metrics(gold, before, row_weights)
    candidate_acc, candidate_f1 = weighted_metrics(gold, after, row_weights)
    groups = {}
    for group in sorted(screen_counts):
        group_ids = [key for key in ids if group_name(baseline[key]) == group]
        helpful = sum(baseline[key]["prediction"] != baseline[key]["gold_label"] and
                      candidate[key]["prediction"] == baseline[key]["gold_label"] for key in group_ids)
        harmful = sum(baseline[key]["prediction"] == baseline[key]["gold_label"] and
                      candidate[key]["prediction"] != baseline[key]["gold_label"] for key in group_ids)
        changed = sum(candidate[key]["prediction"] != baseline[key]["prediction"] for key in group_ids)
        weight = population_counts[group] / screen_counts[group]
        groups[group] = {
            "screen_n": len(group_ids), "population_n": population_counts[group],
            "population_weight_per_screen_row": weight, "helpful": helpful,
            "harmful": harmful, "changed": changed, "net": helpful - harmful,
            "weighted_helpful": helpful * weight, "weighted_harmful": harmful * weight,
            "weighted_net": (helpful - harmful) * weight,
        }
    helpful = sum(item["helpful"] for item in groups.values())
    harmful = sum(item["harmful"] for item in groups.values())
    weighted_helpful = sum(item["weighted_helpful"] for item in groups.values())
    weighted_harmful = sum(item["weighted_harmful"] for item in groups.values())
    return {
        "screen_n": len(ids), "helpful": helpful, "harmful": harmful,
        "changed": sum(item["changed"] for item in groups.values()), "net": helpful - harmful,
        "weighted_helpful": weighted_helpful, "weighted_harmful": weighted_harmful,
        "weighted_net": weighted_helpful - weighted_harmful,
        "projected_baseline_accuracy": base_acc, "projected_candidate_accuracy": candidate_acc,
        "projected_delta_accuracy": candidate_acc - base_acc,
        "projected_baseline_macro_f1": base_f1, "projected_candidate_macro_f1": candidate_f1,
        "projected_delta_macro_f1": candidate_f1 - base_f1,
        "groups": groups,
    }


def summarize_screen(baseline_rows, population_rows, named_candidates):
    baseline = indexed(baseline_rows, "screen baseline")
    population = indexed(population_rows, "population baseline")
    population_counts, screen_counts = {}, {}
    for row in population.values():
        population_counts[group_name(row)] = population_counts.get(group_name(row), 0) + 1
    for row in baseline.values():
        screen_counts[group_name(row)] = screen_counts.get(group_name(row), 0) + 1
    missing_groups = set(population_counts) - set(screen_counts)
    if missing_groups:
        raise ValueError(f"Screen does not represent population groups: {sorted(missing_groups)}")
    actions = {
        action: summarize_action(baseline, indexed(rows, action), population_counts, screen_counts)
        for action, rows in named_candidates
    }
    ranking = sorted(actions, key=lambda action: (
        -actions[action]["weighted_net"], -actions[action]["projected_delta_macro_f1"], action
    ))
    return {
        "evaluation": "transition_stratified_margin_spread_action_screen",
        "population_examples": len(population), "screen_examples": len(baseline),
        "population_group_counts": population_counts, "screen_group_counts": screen_counts,
        "ranking_by_weighted_net": ranking, "actions": actions,
        "limitations": [
            "The screen is deterministic and margin-spanning, not a random probability sample.",
            "Population-weighted metrics are descriptive projections for action screening, not holdout estimates.",
            "No action, router, or threshold is selected by this report alone.",
        ],
    }


def parse_candidate(specification):
    action, separator, filename = specification.partition("=")
    if not separator:
        raise ValueError(f"Expected ACTION=PATH, got {specification}")
    return action, list(read_jsonl(filename))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--population-records", required=True, type=Path)
    parser.add_argument("--candidate", nargs="+", required=True, help="ACTION=predictions.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = summarize_screen(
        read_jsonl(args.baseline_records), read_jsonl(args.population_records),
        [parse_candidate(item) for item in args.candidate],
    )
    atomic_write_json(args.output, report)
    print("ACTION\tHELPFUL\tHARMFUL\tNET\tWEIGHTED_NET\tDELTA_ACC\tDELTA_F1")
    for action in report["ranking_by_weighted_net"]:
        row = report["actions"][action]
        print(f"{action}\t{row['helpful']}\t{row['harmful']}\t{row['net']}\t"
              f"{row['weighted_net']:.3f}\t{row['projected_delta_accuracy']:.6f}\t"
              f"{row['projected_delta_macro_f1']:.6f}")
    print(f"V3_ACTION_SCREEN_SUMMARY_DONE output={args.output}")


if __name__ == "__main__":
    main()
