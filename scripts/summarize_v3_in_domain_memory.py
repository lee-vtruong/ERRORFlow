"""Print compact, copy-safe summaries for in-domain memory experiments."""

import argparse
import json
from pathlib import Path


def compact_metrics(row):
    return {
        key: row[key]
        for key in (
            "routed", "route_rate", "helpful", "harmful", "net",
            "accuracy", "macro_f1", "avg_model_forward_passes",
        )
        if key in row
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.summaries:
        report = json.loads(path.read_text(encoding="utf-8"))
        print(f"\nSUMMARY {path}")
        print("evaluation:", report.get("evaluation"))
        print("dataset:", report.get("dataset", "external"))
        results = report.get("results", {})
        if "natural" in results or "anchors" in results:
            for mode in ("anchors", "natural", "matched_nlf_budget"):
                for policy, row in results.get(mode, {}).items():
                    print(mode, policy, json.dumps(compact_metrics(row), sort_keys=True))
        else:
            for policy, row in results.items():
                print(policy, json.dumps(compact_metrics(row), sort_keys=True))
        if report.get("hypothesis_test"):
            print("hypothesis_test", json.dumps(report["hypothesis_test"], sort_keys=True))
        for name, comparison in report.get("comparisons", {}).items():
            print(name, json.dumps(comparison, sort_keys=True))
        if report.get("fold_results"):
            print("fold_results", json.dumps(report["fold_results"], sort_keys=True))


if __name__ == "__main__":
    main()
