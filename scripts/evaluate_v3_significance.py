"""Paired bootstrap and exact McNemar test for a materialized V3 policy."""

import argparse
import json
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap


INTERPRETATIONS = {
    "train-internal": "train-internal fold-excluded comparison; not an independent holdout",
    "external-source-only": (
        "independent cross-domain evaluation with source-only memory and a frozen policy; "
        "target labels were not used for routing or model selection"
    ),
}


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--evaluation-context",
        choices=sorted(INTERPRETATIONS),
        default="train-internal",
    )
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
    result = {
        "n": len(ids), "baseline_accuracy": accuracy(before, expected),
        "candidate_accuracy": accuracy(after, expected),
        "delta_accuracy": accuracy(after, expected) - accuracy(before, expected),
        "baseline_macro_f1": macro_f1(before, expected),
        "candidate_macro_f1": macro_f1(after, expected),
        "delta_macro_f1": macro_f1(after, expected) - macro_f1(before, expected),
        "mcnemar_exact": mcnemar_exact(expected, before, after),
        "paired_bootstrap": paired_bootstrap(expected, before, after, args.samples, args.seed),
        "evaluation_context": args.evaluation_context,
        "interpretation": INTERPRETATIONS[args.evaluation_context],
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
