"""Print compact, paper-ready summaries from frozen R6-NLF evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def metric_line(name, row):
    return (
        f"{name}\trouted={row['routed']}\trate={row['route_rate']:.6f}"
        f"\thelpful={row['helpful']}\tharmful={row['harmful']}\tnet={row['net']}"
        f"\taccuracy={row['accuracy']:.6f}\tmacro_f1={row['macro_f1']:.6f}"
        f"\tpasses={row['avg_model_forward_passes']:.6f}"
    )


def print_comparison(name, row):
    mcnemar = row["mcnemar_exact"]
    bootstrap = row["paired_bootstrap"]
    print(
        name,
        f"orientation={row['orientation']}",
        f"mcnemar_p={mcnemar['exact_two_sided_p']:.9g}",
        f"delta_accuracy={bootstrap['delta_accuracy']:.6f}",
        f"accuracy_ci95={bootstrap['accuracy_delta_ci95']}",
        f"delta_macro_f1={bootstrap['delta_macro_f1']:.6f}",
        f"macro_f1_ci95={bootstrap['macro_f1_delta_ci95']}",
        sep="\t",
    )


def summarize(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"\nSUMMARY\t{path}")
    print(f"config_id\t{report['config_id']}")
    for mode in ("anchors", "natural", "matched_nlf_budget", "matched_hard_budget"):
        print(f"\nMODE\t{mode}")
        for policy, row in report["results"][mode].items():
            print(metric_line(policy, row))
            f1 = {label: round(values["f1"], 6) for label, values in row["per_class"].items()}
            print(f"PER_CLASS\t{policy}\t{json.dumps(f1, sort_keys=True)}")
    print("\nROUTE_DIFFERENCE")
    print(json.dumps(report["route_difference"], sort_keys=True))
    print("\nNLF_NEIGHBORS")
    print(json.dumps(report["nlf_neighbor_analysis"], sort_keys=True))
    print("\nRETRIEVAL_OVERHEAD")
    print(json.dumps(report["retrieval_overhead"], sort_keys=True))
    print("\nPAIRED_COMPARISONS")
    for name, row in report["comparisons"].items():
        print_comparison(name, row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.summaries:
        summarize(path)


if __name__ == "__main__":
    main()
