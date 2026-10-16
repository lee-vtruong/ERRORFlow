"""Summarize how much Alt benefit a selective router retains and harm it avoids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def label(row):
    return str(row.get("label", row.get("gold_label")))


def filtered_predictions(path, policy=None, mode=None):
    rows = list(read_jsonl(path))
    if policy is not None:
        rows = [row for row in rows if row.get("policy") == policy]
    if mode is not None:
        rows = [row for row in rows if row.get("mode") == mode]
    return indexed(rows, str(path))


def change_counts(gold, base, prediction):
    helpful = sum(b != g and p == g for g, b, p in zip(gold, base, prediction))
    harmful = sum(b == g and p != g for g, b, p in zip(gold, base, prediction))
    return helpful, harmful


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--router-predictions", required=True, type=Path)
    parser.add_argument("--router-policy")
    parser.add_argument("--router-mode")
    parser.add_argument("--project-conflicting-to-nei", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows-output", type=Path)
    args = parser.parse_args()

    gold_rows = indexed(read_jsonl(args.gold), "gold")
    baseline_rows = indexed(read_jsonl(args.baseline), "baseline")
    candidate_rows = indexed(read_jsonl(args.candidate), "candidate")
    router_rows = filtered_predictions(
        args.router_predictions, args.router_policy, args.router_mode
    )
    ids = sorted(gold_rows)
    for name, rows in (
        ("baseline", baseline_rows), ("candidate", candidate_rows), ("router", router_rows)
    ):
        if set(rows) != set(ids):
            raise ValueError(f"{name} IDs do not exactly match gold IDs")

    expected = [label(gold_rows[key]) for key in ids]
    base = [str(baseline_rows[key]["prediction"]) for key in ids]
    alt = [str(candidate_rows[key]["prediction"]) for key in ids]
    router = [str(router_rows[key]["prediction"]) for key in ids]
    if args.project_conflicting_to_nei:
        project = lambda value: "NOT ENOUGH INFO" if value == "CONFLICTING" else value
        base, alt, router = map(lambda values: list(map(project, values)), (base, alt, router))

    alt_helpful, alt_harmful = change_counts(expected, base, alt)
    router_helpful, router_harmful = change_counts(expected, base, router)
    routed_flags = [
        bool(router_rows[key]["routed"])
        if "routed" in router_rows[key]
        else router[index] != base[index]
        for index, key in enumerate(ids)
    ]
    routed = sum(routed_flags)
    report = {
        "evaluation": "selective_escalation_harm_containment",
        "setting": args.name,
        "n": len(ids),
        "router_filter": {"policy": args.router_policy, "mode": args.router_mode},
        "projection_conflicting_to_nei": args.project_conflicting_to_nei,
        "results": {
            "Base": {"accuracy": accuracy(base, expected), "macro_f1": macro_f1(base, expected)},
            "Alt-All": {
                "accuracy": accuracy(alt, expected), "macro_f1": macro_f1(alt, expected),
                "helpful": alt_helpful, "harmful": alt_harmful,
            },
            "Router": {
                "accuracy": accuracy(router, expected), "macro_f1": macro_f1(router, expected),
                "routed": routed, "route_rate": routed / len(ids),
                "helpful": router_helpful, "harmful": router_harmful,
            },
        },
        "harm_containment": {
            "harm_avoided_count": alt_harmful - router_harmful,
            "harm_avoided_fraction": (
                1.0 - router_harmful / alt_harmful if alt_harmful else None
            ),
            "helpful_retained_count": router_helpful,
            "helpful_retained_fraction": (
                router_helpful / alt_helpful if alt_helpful else None
            ),
            "interpretation": (
                "Descriptive containment relative to invoking the same Alt on every example; "
                "not a formal no-harm guarantee."
            ),
        },
    }
    rows = [{
        "example_id": key,
        "gold_label": expected[index],
        "baseline_prediction": base[index],
        "candidate_prediction": alt[index],
        "router_prediction": router[index],
        "routed": routed_flags[index],
        "alt_change": "helpful" if base[index] != expected[index] and alt[index] == expected[index]
        else "harmful" if base[index] == expected[index] and alt[index] != expected[index]
        else "neutral",
        "router_change": "helpful" if base[index] != expected[index] and router[index] == expected[index]
        else "harmful" if base[index] == expected[index] and router[index] != expected[index]
        else "neutral",
    } for index, key in enumerate(ids)]
    atomic_write_json(args.output, report)
    if args.rows_output:
        write_jsonl(args.rows_output, rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
