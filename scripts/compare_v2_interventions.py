"""Compare intervention utility by train-only screening group."""

import argparse
import json
from pathlib import Path

from errorflow.io import atomic_write_json


def compare(named_reports):
    reports = {name: report for name, report in named_reports}
    groups = sorted({group for report in reports.values() for group in report["by_group"]})
    matrix = {}
    for group in groups:
        actions = {}
        for action, report in reports.items():
            stats = report["by_group"].get(group, {"n": 0, "recovered": 0, "regressed": 0, "changed": 0})
            n = stats["n"]
            actions[action] = {
                **stats,
                "recovery_rate": stats["recovered"] / n if n else 0.0,
                "regression_rate": stats["regressed"] / n if n else 0.0,
                "net_utility_rate": (stats["recovered"] - stats["regressed"]) / n if n else 0.0,
            }
        ranked = sorted(actions, key=lambda action: (-actions[action]["net_utility_rate"], actions[action]["changed"], action))
        matrix[group] = {"best_action": ranked[0] if ranked else None, "actions": actions}
    aggregate = {
        action: {key: report[key] for key in (
            "n", "helpful", "harmful", "net_correct_change", "baseline_accuracy",
            "candidate_accuracy", "candidate_macro_f1", "avg_candidate_latency_ms",
            "avg_candidate_forward_passes",
        )}
        for action, report in reports.items()
    }
    return {"aggregate": aggregate, "by_group": matrix}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True, help="ACTION=metrics.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    named = []
    for specification in args.metrics:
        action, separator, filename = specification.partition("=")
        if not separator:
            raise ValueError(f"Expected ACTION=PATH, got {specification}")
        named.append((action, json.loads(Path(filename).read_text(encoding="utf-8"))))
    result = compare(named)
    atomic_write_json(args.output, result)
    print("GROUP\tACTION\tN\tRECOVERED\tREGRESSED\tCHANGED\tUTILITY_RATE")
    for group, row in result["by_group"].items():
        for action, stats in row["actions"].items():
            print(f"{group}\t{action}\t{stats['n']}\t{stats['recovered']}\t{stats['regressed']}\t{stats['changed']}\t{stats['net_utility_rate']:.4f}")
    print(f"COMPARISON_DONE output={args.output}")


if __name__ == "__main__":
    main()
