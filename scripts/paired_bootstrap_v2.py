"""Paired bootstrap for canonical V2 prediction artifacts."""

import argparse
import json
import random
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


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    gold = indexed(read_jsonl(args.gold), "gold")
    baseline = indexed(read_jsonl(args.baseline), "baseline")
    candidate = indexed(read_jsonl(args.candidate), "candidate")
    if set(gold) != set(baseline) or set(gold) != set(candidate):
        raise ValueError("Gold, baseline, and candidate IDs must match exactly")
    ids = sorted(gold)
    expected = [gold[key]["label"] for key in ids]
    before = [baseline[key]["prediction"] for key in ids]
    after = [candidate[key]["prediction"] for key in ids]
    base_accuracy, candidate_accuracy = accuracy(before, expected), accuracy(after, expected)
    base_f1, candidate_f1 = macro_f1(before, expected), macro_f1(after, expected)
    helpful = sum(b != g and a == g for g, b, a in zip(expected, before, after))
    harmful = sum(b == g and a != g for g, b, a in zip(expected, before, after))
    rng = random.Random(args.seed)
    accuracy_deltas, f1_deltas = [], []
    for _ in range(args.samples):
        sample = [rng.randrange(len(ids)) for _ in ids]
        sample_gold = [expected[index] for index in sample]
        sample_before = [before[index] for index in sample]
        sample_after = [after[index] for index in sample]
        accuracy_deltas.append(accuracy(sample_after, sample_gold) - accuracy(sample_before, sample_gold))
        f1_deltas.append(macro_f1(sample_after, sample_gold) - macro_f1(sample_before, sample_gold))
    result = {
        "n": len(ids), "samples": args.samples, "seed": args.seed,
        "helpful": helpful, "harmful": harmful, "net": helpful - harmful,
        "baseline_accuracy": base_accuracy, "candidate_accuracy": candidate_accuracy,
        "delta_accuracy": candidate_accuracy - base_accuracy,
        "accuracy_ci95": [percentile(accuracy_deltas, 0.025), percentile(accuracy_deltas, 0.975)],
        "p_delta_accuracy_gt_0": sum(value > 0 for value in accuracy_deltas) / len(accuracy_deltas),
        "baseline_macro_f1": base_f1, "candidate_macro_f1": candidate_f1,
        "delta_macro_f1": candidate_f1 - base_f1,
        "macro_f1_ci95": [percentile(f1_deltas, 0.025), percentile(f1_deltas, 0.975)],
        "p_delta_macro_f1_gt_0": sum(value > 0 for value in f1_deltas) / len(f1_deltas),
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
