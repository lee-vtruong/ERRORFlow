"""Select same-label versus global-neighbor compatibility by source cross-fit.

The input policies must already be materialized with group-safe fold exclusion.
For every held-out fold, the rule is selected using only the other folds'
labels.  A separate all-source choice is exported for frozen deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap


POLICIES = {
    "Memory-SameLabel": "D0_HardLabel",
    "Memory-Global": "D1_NLF",
}


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def gold_label(row):
    value = row.get("label", row.get("gold_label"))
    if value is None:
        raise ValueError(f"Missing gold label for {row.get('example_id')}")
    return str(value)


def metrics(ids, gold, predictions):
    expected = [gold_label(gold[key]) for key in ids]
    predicted = [predictions[key]["prediction"] for key in ids]
    return {
        "n": len(ids),
        "routed": sum(bool(predictions[key].get("routed")) for key in ids),
        "accuracy": accuracy(predicted, expected),
        "macro_f1": macro_f1(predicted, expected),
    }


def choose_rule(ids, gold, policies, criterion):
    scored = []
    for name in sorted(policies):
        result = metrics(ids, gold, policies[name])
        primary = result[criterion]
        secondary = result["macro_f1" if criterion == "accuracy" else "accuracy"]
        # The lexical final field makes ties deterministic.  Same-label is the
        # conservative default only when both reported metrics are identical.
        tie_preference = int(name == "Memory-SameLabel")
        scored.append((primary, secondary, tie_preference, name, result))
    return max(scored)[3], {item[3]: item[4] for item in scored}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--meta-dir", required=True, type=Path)
    parser.add_argument("--criterion", choices=("accuracy", "macro_f1"), default="macro_f1")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    gold = indexed(read_jsonl(args.gold), "gold")
    feature_rows = indexed(
        read_jsonl(args.meta_dir / "per_query_features.jsonl"), "meta features"
    )
    ids = sorted(gold)
    if set(feature_rows) != set(ids):
        raise ValueError("Gold and meta feature IDs do not match")
    folds = {key: int(feature_rows[key]["outer_fold"]) for key in ids}

    all_predictions = list(read_jsonl(args.meta_dir / "per_query_predictions.jsonl"))
    policies = {}
    for semantic_name, artifact_name in POLICIES.items():
        rows = [
            row for row in all_predictions
            if row.get("mode") == "natural" and row.get("policy") == artifact_name
        ]
        policies[semantic_name] = indexed(rows, semantic_name)
        if set(policies[semantic_name]) != set(ids):
            raise ValueError(f"{semantic_name} prediction IDs do not match gold")

    selected_predictions = {}
    fold_reports = []
    for held_out in sorted(set(folds.values())):
        train_ids = [key for key in ids if folds[key] != held_out]
        test_ids = [key for key in ids if folds[key] == held_out]
        selected, train_metrics = choose_rule(
            train_ids, gold, policies, args.criterion
        )
        for key in test_ids:
            selected_predictions[key] = policies[selected][key]["prediction"]
        fold_reports.append({
            "outer_fold": held_out,
            "selection_examples": len(train_ids),
            "evaluation_examples": len(test_ids),
            "selected_rule": selected,
            "selection_metrics": train_metrics,
            "evaluation_metrics": metrics(test_ids, gold, policies[selected]),
            "held_out_labels_used_for_selection": False,
        })

    global_rule, global_metrics = choose_rule(ids, gold, policies, args.criterion)
    expected = [gold_label(gold[key]) for key in ids]
    selected = [selected_predictions[key] for key in ids]
    selected_metrics = {
        "n": len(ids),
        "accuracy": accuracy(selected, expected),
        "macro_f1": macro_f1(selected, expected),
    }
    comparisons = {}
    for name, rows in policies.items():
        candidate = [rows[key]["prediction"] for key in ids]
        comparisons[f"CrossFitSelection_vs_{name}"] = {
            "orientation": "crossfit_selection_minus_fixed_rule",
            "mcnemar_exact": mcnemar_exact(expected, candidate, selected),
            "paired_bootstrap": paired_bootstrap(
                expected, candidate, selected, args.bootstrap_samples, args.seed
            ),
        }

    report = {
        "evaluation": "group_safe_compatibility_rule_selection",
        "criterion": args.criterion,
        "n": len(ids),
        "candidate_rules": list(POLICIES),
        "folds": fold_reports,
        "cross_fitted_selection": selected_metrics,
        "all_source_deployment_choice": {
            "selected_rule": global_rule,
            "source_oof_metrics": global_metrics,
            "target_labels_required_for_deployment": False,
        },
        "comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", report)
    write_jsonl(args.output_dir / "predictions.jsonl", [
        {
            "example_id": key,
            "outer_fold": folds[key],
            "selected_rule": next(
                row["selected_rule"] for row in fold_reports
                if row["outer_fold"] == folds[key]
            ),
            "prediction": selected_predictions[key],
            "gold_label": gold_label(gold[key]),
        }
        for key in ids
    ])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
