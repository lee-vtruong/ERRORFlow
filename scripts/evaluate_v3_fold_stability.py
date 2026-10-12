"""Summarize a materialized V3 policy across preassigned outer folds."""

import argparse
import json
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def summarize_folds(baseline_rows, candidate_rows):
    baseline = indexed(baseline_rows, "baseline")
    candidate = indexed(candidate_rows, "candidate")
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate IDs must match exactly")
    folds = sorted({row.get("outer_fold") for row in baseline.values()})
    if None in folds:
        raise ValueError("Every baseline row must have an outer_fold")
    results = []
    for fold in folds:
        ids = sorted(key for key, row in baseline.items() if row["outer_fold"] == fold)
        gold = [baseline[key]["gold_label"] for key in ids]
        before = [baseline[key]["prediction"] for key in ids]
        after = [candidate[key]["prediction"] for key in ids]
        helpful = sum(b != g and a == g for g, b, a in zip(gold, before, after))
        harmful = sum(b == g and a != g for g, b, a in zip(gold, before, after))
        base_accuracy, final_accuracy = accuracy(before, gold), accuracy(after, gold)
        base_f1, final_f1 = macro_f1(before, gold), macro_f1(after, gold)
        routed = sum(bool(candidate[key].get("routed")) for key in ids)
        results.append({
            "outer_fold": fold, "n": len(ids), "routed": routed,
            "route_rate": routed / len(ids), "helpful": helpful,
            "harmful": harmful, "net": helpful - harmful,
            "baseline_accuracy": base_accuracy, "candidate_accuracy": final_accuracy,
            "delta_accuracy": final_accuracy - base_accuracy,
            "baseline_macro_f1": base_f1, "candidate_macro_f1": final_f1,
            "delta_macro_f1": final_f1 - base_f1,
        })
    return {
        "n_folds": len(results), "folds": results,
        "positive_accuracy_folds": sum(row["delta_accuracy"] > 0 for row in results),
        "positive_macro_f1_folds": sum(row["delta_macro_f1"] > 0 for row in results),
        "mean_fold_delta_accuracy": sum(row["delta_accuracy"] for row in results) / len(results),
        "mean_fold_delta_macro_f1": sum(row["delta_macro_f1"] for row in results) / len(results),
        "min_fold_delta_accuracy": min(row["delta_accuracy"] for row in results),
        "min_fold_delta_macro_f1": min(row["delta_macro_f1"] for row in results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = summarize_folds(read_jsonl(args.baseline_records), read_jsonl(args.candidate))
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
