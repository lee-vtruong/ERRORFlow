"""Group-bootstrap any two cached prediction systems on the same claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl
from errorflow.v3_statistics import grouped_paired_bootstrap, mcnemar_exact, paired_bootstrap

try:
    from scripts.evaluate_v3_grouping_sensitivity import grouped_folds, near_duplicate_edges
except ModuleNotFoundError:
    from evaluate_v3_grouping_sensitivity import grouped_folds, near_duplicate_edges


def parse_filters(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Filter must be KEY=VALUE: {value}")
        key, expected = value.split("=", 1)
        result[key] = expected
    return result


def indexed_predictions(path, filters, name):
    rows = [
        row for row in read_jsonl(path)
        if all(str(row.get(key)) == value for key, value in filters.items())
    ]
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = str(row["prediction"])
    if not result:
        raise ValueError(f"No rows selected for {name} with filters {filters}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-filter", action="append", default=[])
    parser.add_argument("--candidate-filter", action="append", default=[])
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--group-threshold", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    gold_rows = list(read_jsonl(args.gold))
    gold = {str(row["example_id"]): str(row.get("label", row.get("gold_label"))) for row in gold_rows}
    before = indexed_predictions(args.baseline, parse_filters(args.baseline_filter), args.baseline_name)
    after = indexed_predictions(args.candidate, parse_filters(args.candidate_filter), args.candidate_name)
    if set(gold) != set(before) or set(gold) != set(after):
        raise ValueError("Gold, baseline, and candidate IDs must match exactly after filtering")
    ids = sorted(gold)
    expected = [gold[key] for key in ids]
    baseline_predictions = [before[key] for key in ids]
    candidate_predictions = [after[key] for key in ids]
    edges = near_duplicate_edges(gold_rows, args.group_threshold)
    _, components = grouped_folds(gold_rows, edges, args.group_threshold, args.folds, args.seed)
    component_by_id = {key: min(component) for component in components for key in component}
    groups = [component_by_id[key] for key in ids]
    report = {
        "evaluation": "grouped_paired_prediction_comparison",
        "orientation": f"{args.candidate_name}_minus_{args.baseline_name}",
        "n": len(ids),
        "grouping": {
            "threshold": args.group_threshold,
            "n_groups": len(components),
            "largest_group": max(map(len, components)),
        },
        "results": {
            args.baseline_name: {
                "accuracy": accuracy(baseline_predictions, expected),
                "macro_f1": macro_f1(baseline_predictions, expected),
            },
            args.candidate_name: {
                "accuracy": accuracy(candidate_predictions, expected),
                "macro_f1": macro_f1(candidate_predictions, expected),
            },
        },
        "comparison": {
            "mcnemar_exact": mcnemar_exact(expected, baseline_predictions, candidate_predictions),
            "claim_bootstrap": paired_bootstrap(
                expected, baseline_predictions, candidate_predictions, args.samples, args.seed
            ),
            "group_bootstrap": grouped_paired_bootstrap(
                expected, baseline_predictions, candidate_predictions, groups, args.samples, args.seed
            ),
        },
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
