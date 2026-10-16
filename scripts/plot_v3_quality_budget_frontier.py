"""Plot protocol-separated quality--compute frontiers from CSV artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DISPLAY = {
    "No-label-filter-memory": "Memory-Global",
    "Confidence": "Confidence",
    "Logistic": "Logistic",
    "Logistic-BalancedOutcome": "Logistic (balanced)",
    "Logistic-PerLabelRank": "Logistic (per-label rank)",
    "Base": "Base",
    "Alt-All": "Alt-All",
    "Score-Ensemble": "Score ensemble",
    "Oracle-at-most-budget": "Oracle upper bound",
}

COLORS = {
    "Memory-Global": "#1f77b4",
    "Confidence": "#ff7f0e",
    "Logistic": "#2ca02c",
    "Logistic (balanced)": "#9467bd",
    "Logistic (per-label rank)": "#8c564b",
    "Base": "#222222",
    "Alt-All": "#d62728",
    "Score ensemble": "#17becf",
    "Oracle upper bound": "#7f7f7f",
}


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_curve(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use DATASET=path/to/quality_budget_curves.csv")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Both dataset name and CSV path are required")
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve", action="append", type=parse_curve, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    n = len(args.curve)
    figure, axes = plt.subplots(n, 2, figsize=(8.2, max(2.3 * n, 3.0)), squeeze=False)
    for row_index, (dataset, path) in enumerate(args.curve):
        rows = read_rows(path)
        for col_index, metric in enumerate(("accuracy", "macro_f1")):
            axis = axes[row_index][col_index]
            for artifact_name, display_name in DISPLAY.items():
                selected = [row for row in rows if row.get("policy") == artifact_name]
                if not selected:
                    continue
                selected.sort(key=lambda row: float(row["avg_model_forward_passes"]))
                x = [float(row["avg_model_forward_passes"]) for row in selected]
                y = [100.0 * float(row[metric]) for row in selected]
                fixed = artifact_name in {"Base", "Alt-All", "Score-Ensemble"}
                axis.plot(
                    x, y, marker="o", markersize=3.5,
                    linewidth=0 if fixed else 1.4,
                    color=COLORS[display_name], label=display_name,
                )
            axis.set_ylabel(f"{dataset}\n{metric.replace('_', '-').title()} (%)" if col_index == 0
                            else f"{metric.replace('_', '-').title()} (%)")
            axis.set_xlabel("Average model forward passes")
            axis.grid(alpha=.25, linewidth=.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    print(f"FRONTIER_FIGURE_DONE datasets={n} output={args.output}")


if __name__ == "__main__":
    main()
