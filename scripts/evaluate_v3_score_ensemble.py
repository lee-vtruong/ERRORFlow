"""Evaluate a supervised score-level B1/A3 ensemble without new LLM calls.

For grouped source evaluation, the mixture weight is selected on complementary
outer folds and applied only to the held-out fold.  A single deployable weight
is selected on the complete source set for frozen external evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap


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
    return value


def label_scores(row):
    scores = row.get("label_scores") or row.get("metadata", {}).get("label_scores")
    if not scores:
        raise ValueError(f"Missing label_scores for {row.get('example_id')}")
    return {str(key): float(value) for key, value in scores.items()}


def probabilities(scores):
    peak = max(scores.values())
    values = {key: math.exp(value - peak) for key, value in scores.items()}
    total = sum(values.values())
    return {key: value / total for key, value in values.items()}


def mixed_prediction(b1_row, a3_row, alpha, project_conflicting=False):
    left, right = probabilities(label_scores(b1_row)), probabilities(label_scores(a3_row))
    labels = sorted(set(left) | set(right))
    mixed = {
        label: alpha * left.get(label, 0.0) + (1.0 - alpha) * right.get(label, 0.0)
        for label in labels
    }
    if project_conflicting and "CONFLICTING" in mixed:
        mixed["NOT ENOUGH INFO"] = mixed.get("NOT ENOUGH INFO", 0.0) + mixed.pop("CONFLICTING")
    prediction = max(mixed, key=lambda label: (mixed[label], label))
    return prediction, mixed


def predictions(ids, baseline, candidate, alpha, project_conflicting=False):
    return [
        mixed_prediction(baseline[key], candidate[key], alpha, project_conflicting)[0]
        for key in ids
    ]


def choose_alpha(ids, gold, baseline, candidate, grid, project_conflicting=False):
    expected = [gold_label(gold[key]) for key in ids]
    scored = []
    for alpha in grid:
        predicted = predictions(ids, baseline, candidate, alpha, project_conflicting)
        scored.append((macro_f1(predicted, expected), accuracy(predicted, expected), alpha))
    return max(scored)[2], [
        {"alpha": alpha, "accuracy": acc, "macro_f1": f1}
        for f1, acc, alpha in sorted(scored, key=lambda item: item[2])
    ]


def metrics(expected, predicted):
    return {"accuracy": accuracy(predicted, expected), "macro_f1": macro_f1(predicted, expected)}


def load_folds(meta_dir):
    rows = read_jsonl(Path(meta_dir) / "per_query_features.jsonl")
    folds = {str(row["example_id"]): row.get("outer_fold") for row in rows}
    if not folds or any(value is None for value in folds.values()):
        raise ValueError("Grouped source evaluation requires outer_fold in meta features")
    return {key: int(value) for key, value in folds.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", choices=("grouped_internal", "external"), required=True)
    parser.add_argument("--source-gold", type=Path)
    parser.add_argument("--source-baseline", type=Path)
    parser.add_argument("--source-candidate", type=Path)
    parser.add_argument("--source-meta-dir", type=Path)
    parser.add_argument("--meta-dir", type=Path)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[i / 20 for i in range(21)])
    parser.add_argument("--project-conflicting-to-nei", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    gold = indexed(read_jsonl(args.gold), "gold")
    baseline = indexed(read_jsonl(args.baseline), "baseline")
    candidate = indexed(read_jsonl(args.candidate), "candidate")
    ids = sorted(gold)
    if set(ids) != set(baseline) or set(ids) != set(candidate):
        raise ValueError("Gold, baseline, and candidate IDs must match")
    if any(not 0.0 <= value <= 1.0 for value in args.alpha_grid):
        raise ValueError("Every alpha must be in [0, 1]")

    selection = {}
    if args.protocol == "grouped_internal":
        if not args.meta_dir:
            parser.error("--meta-dir is required for grouped_internal")
        folds = load_folds(args.meta_dir)
        if set(folds) != set(ids):
            raise ValueError("Meta feature IDs do not match source IDs")
        final = {}
        for held_out in sorted(set(folds.values())):
            train_ids = [key for key in ids if folds[key] != held_out]
            test_ids = [key for key in ids if folds[key] == held_out]
            alpha, grid = choose_alpha(
                train_ids, gold, baseline, candidate, args.alpha_grid,
                args.project_conflicting_to_nei,
            )
            selection[str(held_out)] = {"alpha": alpha, "training_n": len(train_ids), "grid": grid}
            for key in test_ids:
                final[key] = mixed_prediction(
                    baseline[key], candidate[key], alpha, args.project_conflicting_to_nei
                )
        deployable_alpha, deployable_grid = choose_alpha(
            ids, gold, baseline, candidate, args.alpha_grid, args.project_conflicting_to_nei
        )
        selection["deployable_source"] = {"alpha": deployable_alpha, "training_n": len(ids), "grid": deployable_grid}
        selection_used_target_labels = False
    else:
        required = (args.source_gold, args.source_baseline, args.source_candidate)
        if not all(required):
            parser.error("external requires --source-gold, --source-baseline, and --source-candidate")
        source_gold = indexed(read_jsonl(args.source_gold), "source gold")
        source_baseline = indexed(read_jsonl(args.source_baseline), "source baseline")
        source_candidate = indexed(read_jsonl(args.source_candidate), "source candidate")
        source_ids = sorted(source_gold)
        if set(source_ids) != set(source_baseline) or set(source_ids) != set(source_candidate):
            raise ValueError("Source IDs do not match")
        deployable_alpha, deployable_grid = choose_alpha(
            source_ids, source_gold, source_baseline, source_candidate,
            args.alpha_grid, False,
        )
        selection["deployable_source"] = {
            "alpha": deployable_alpha, "training_n": len(source_ids), "grid": deployable_grid,
        }
        final = {
            key: mixed_prediction(
                baseline[key], candidate[key], deployable_alpha, args.project_conflicting_to_nei
            ) for key in ids
        }
        selection_used_target_labels = False

    expected = [gold_label(gold[key]) for key in ids]
    b1 = [baseline[key]["prediction"] for key in ids]
    a3 = [candidate[key]["prediction"] for key in ids]
    if args.project_conflicting_to_nei:
        b1 = ["NOT ENOUGH INFO" if value == "CONFLICTING" else value for value in b1]
        a3 = ["NOT ENOUGH INFO" if value == "CONFLICTING" else value for value in a3]
    ensemble = [final[key][0] for key in ids]
    rows = []
    for key, prediction in zip(ids, ensemble):
        rows.append({
            "example_id": key, "gold_label": gold_label(gold[key]),
            "baseline_prediction": baseline[key]["prediction"],
            "candidate_prediction": candidate[key]["prediction"],
            "prediction": prediction, "mixed_probabilities": final[key][1],
            "alpha": selection[str(folds[key])]["alpha"] if args.protocol == "grouped_internal" else deployable_alpha,
        })
    report = {
        "evaluation": "score_level_b1_a3_ensemble",
        "protocol": args.protocol,
        "n": len(ids),
        "source_labels_used_for_weight_selection": True,
        "evaluation_example_labels_used_for_own_weight_selection": selection_used_target_labels,
        "compute": {"model_forward_passes_per_example": 8.0, "routes": len(ids)},
        "results": {"B1": metrics(expected, b1), "A3-All": metrics(expected, a3), "Score-Ensemble": metrics(expected, ensemble)},
        "comparisons": {
            "ensemble_vs_b1": {
                "mcnemar_exact": mcnemar_exact(expected, b1, ensemble),
                "paired_bootstrap": paired_bootstrap(expected, b1, ensemble, args.bootstrap_samples, args.seed),
            },
            "ensemble_vs_a3": {
                "mcnemar_exact": mcnemar_exact(expected, a3, ensemble),
                "paired_bootstrap": paired_bootstrap(expected, a3, ensemble, args.bootstrap_samples, args.seed),
            },
        },
        "weight_selection": selection,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    atomic_write_json(args.output_dir / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
