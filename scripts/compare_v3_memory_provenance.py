"""Paired comparison of two routers on the same claims, including group bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_statistics import grouped_paired_bootstrap, mcnemar_exact, paired_bootstrap

try:
    from scripts.evaluate_v3_grouping_sensitivity import grouped_folds, near_duplicate_edges
except ModuleNotFoundError:
    from evaluate_v3_grouping_sensitivity import grouped_folds, near_duplicate_edges


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def select(path, policy=None, mode=None):
    rows = list(read_jsonl(path))
    if policy is not None:
        rows = [row for row in rows if row.get("policy") == policy]
    if mode is not None:
        rows = [row for row in rows if row.get("mode") == mode]
    return indexed(rows, str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--in-domain-predictions", required=True, type=Path)
    parser.add_argument("--in-domain-policy", default="Memory-Global-natural")
    parser.add_argument("--source-memory-predictions", required=True, type=Path)
    parser.add_argument("--source-memory-policy", default="D1_NLF")
    parser.add_argument("--source-memory-mode", default="natural")
    parser.add_argument("--group-threshold", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows-output", type=Path)
    args = parser.parse_args()

    gold_rows = list(read_jsonl(args.gold))
    gold = indexed(gold_rows, "gold")
    left = select(args.source_memory_predictions, args.source_memory_policy, args.source_memory_mode)
    right = select(args.in_domain_predictions, args.in_domain_policy, None)
    ids = sorted(gold)
    if set(left) != set(ids) or set(right) != set(ids):
        raise ValueError("Both router outputs must contain exactly the gold example IDs")
    expected = [str(gold[key].get("label", gold[key].get("gold_label"))) for key in ids]
    source_predictions = [str(left[key]["prediction"]) for key in ids]
    in_domain_predictions = [str(right[key]["prediction"]) for key in ids]

    edges = near_duplicate_edges(gold_rows, args.group_threshold)
    _, components = grouped_folds(gold_rows, edges, args.group_threshold, args.folds, args.seed)
    component_by_id = {key: min(component) for component in components for key in component}
    groups = [component_by_id[key] for key in ids]
    report = {
        "evaluation": "paired_memory_provenance_comparison",
        "orientation": "in_domain_minus_source_memory",
        "n": len(ids),
        "grouping": {
            "threshold": args.group_threshold,
            "n_groups": len(components),
            "largest_group": max(map(len, components)),
        },
        "results": {
            "source_memory": {
                "accuracy": accuracy(source_predictions, expected),
                "macro_f1": macro_f1(source_predictions, expected),
                "routed": sum(bool(left[key].get("routed")) for key in ids),
            },
            "in_domain_memory": {
                "accuracy": accuracy(in_domain_predictions, expected),
                "macro_f1": macro_f1(in_domain_predictions, expected),
                "routed": sum(bool(right[key].get("routed")) for key in ids),
            },
        },
        "comparison": {
            "delta_accuracy": accuracy(in_domain_predictions, expected) - accuracy(source_predictions, expected),
            "delta_macro_f1": macro_f1(in_domain_predictions, expected) - macro_f1(source_predictions, expected),
            "mcnemar_exact": mcnemar_exact(expected, source_predictions, in_domain_predictions),
            "claim_bootstrap": paired_bootstrap(
                expected, source_predictions, in_domain_predictions, args.samples, args.seed
            ),
            "group_bootstrap": grouped_paired_bootstrap(
                expected, source_predictions, in_domain_predictions, groups, args.samples, args.seed
            ),
        },
    }
    atomic_write_json(args.output, report)
    if args.rows_output:
        write_jsonl(args.rows_output, ({
            "example_id": key,
            "group_id": component_by_id[key],
            "gold_label": expected[index],
            "source_memory_prediction": source_predictions[index],
            "in_domain_memory_prediction": in_domain_predictions[index],
            "source_memory_routed": bool(left[key].get("routed")),
            "in_domain_memory_routed": bool(right[key].get("routed")),
        } for index, key in enumerate(ids)))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
