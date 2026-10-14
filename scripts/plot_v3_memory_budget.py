"""Plot the B7/B9 matched-budget accuracy trade-off from the frozen summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def series(summary, method):
    rows = [
        value for key, value in summary["budget_results"].items()
        if key.startswith(f"{method}:budget=")
    ]
    return sorted(rows, key=lambda row: row["budget"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    report = json.loads(args.summary.read_text(encoding="utf-8"))
    full = series(report, "B9_full_outcome_memory")
    error = series(report, "B7_error_only")
    if not full or [row["budget"] for row in full] != [row["budget"] for row in error]:
        raise ValueError("B7 and B9 must have the same non-empty budget grid")

    budgets = [100 * row["budget"] for row in full]
    fig, axis = plt.subplots(figsize=(5.2, 3.2))
    axis.plot(
        budgets, [100 * row["accuracy"] for row in full],
        marker="o", linewidth=2, label="Full outcome memory (B9)",
    )
    axis.plot(
        budgets, [100 * row["accuracy"] for row in error],
        marker="s", linewidth=2, label="Error-only memory (B7)",
    )
    if any(abs(value - 11.18) < 0.01 for value in budgets):
        axis.axvline(11.18, color="0.55", linestyle="--", linewidth=1)
    axis.set_xlabel("Correction budget (%)")
    axis.set_ylabel("Accuracy (%)")
    axis.set_xticks(budgets, [f"{value:g}" for value in budgets], rotation=35)
    values = [100 * row["accuracy"] for row in full + error]
    axis.set_ylim(min(values) - 0.2, max(values) + 0.12)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"MEMORY_BUDGET_FIGURE_DONE output={args.output}")


if __name__ == "__main__":
    main()
