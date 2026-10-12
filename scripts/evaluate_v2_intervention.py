"""Measure train-only recovery, regression, and cost for an intervention."""

import argparse
import json
from collections import defaultdict
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


def evaluate(gold_rows, baseline_rows, candidate_rows):
    gold, baseline, candidate = indexed(gold_rows, "gold"), indexed(baseline_rows, "baseline"), indexed(candidate_rows, "candidate")
    ids = sorted(gold)
    missing = set(ids) - set(candidate)
    if missing:
        raise ValueError(f"Candidate predictions missing {len(missing)} screening IDs")
    by_group = defaultdict(lambda: {"n": 0, "recovered": 0, "regressed": 0, "changed": 0})
    helpful = harmful = changed = baseline_correct = candidate_correct = 0
    expected_labels, baseline_labels, candidate_labels = [], [], []
    for key in ids:
        expected, before, after = gold[key]["label"], baseline[key]["prediction"], candidate[key]["prediction"]
        expected_labels.append(expected); baseline_labels.append(before); candidate_labels.append(after)
        before_ok, after_ok = before == expected, after == expected
        baseline_correct += before_ok; candidate_correct += after_ok
        helpful += (not before_ok and after_ok); harmful += (before_ok and not after_ok); changed += before != after
        group = "control:" + expected if before_ok else f"error:{expected}->{before}"
        stats = by_group[group]; stats["n"] += 1; stats["recovered"] += (not before_ok and after_ok)
        stats["regressed"] += (before_ok and not after_ok); stats["changed"] += before != after
    return {
        "n": len(ids), "baseline_correct": baseline_correct, "candidate_correct": candidate_correct,
        "helpful": helpful, "harmful": harmful, "net_correct_change": helpful - harmful, "changed": changed,
        "baseline_accuracy": accuracy(baseline_labels, expected_labels), "candidate_accuracy": accuracy(candidate_labels, expected_labels),
        "baseline_macro_f1": macro_f1(baseline_labels, expected_labels), "candidate_macro_f1": macro_f1(candidate_labels, expected_labels),
        "avg_candidate_latency_ms": sum(float(candidate[k].get("latency_ms", 0)) for k in ids) / len(ids) if ids else 0.0,
        "avg_candidate_forward_passes": sum(float(candidate[k].get("metadata", {}).get("model_forward_passes", 1)) for k in ids) / len(ids) if ids else 0.0,
        "by_group": dict(sorted(by_group.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(read_jsonl(args.gold), read_jsonl(args.baseline), read_jsonl(args.candidate))
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
