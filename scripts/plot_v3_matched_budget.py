"""Create paper-oriented PDF figures for the V3 matched-budget ablation."""

import argparse
import json
from pathlib import Path


DISPLAY = {
    "B1": "B1", "A3-All": "A3-All", "R-Random": "Random",
    "R-Conf": "Confidence", "R-Margin": "Margin",
    "R-Label": "Label conditional", "R-Sim": "Similarity only",
    "R-Static": "Static pattern", "R-Memory": "Residual memory",
    "R6": "R6 utility",
}
COLORS = {
    "B1": "#4C566A", "A3-All": "#A3A3A3", "R-Random": "#9CA3AF",
    "R-Conf": "#F59E0B", "R-Margin": "#D97706", "R-Label": "#8B5CF6",
    "R-Sim": "#EC4899", "R-Static": "#14B8A6", "R-Memory": "#2563EB",
    "R6": "#DC2626",
}


def configure_plotting():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required: python -m pip install matplotlib") from exc
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
    })
    return plt


def deterministic_rows(summary):
    return [row for row in summary["results"] if "accuracy" in row]


def natural_rows(summary):
    budget = summary["natural_r6_budget"]
    rows = {
        row["policy"]: row for row in deterministic_rows(summary)
        if abs(row["requested_budget"] - budget) < 1e-12
    }
    rows["B1"] = next(row for row in deterministic_rows(summary) if row["policy"] == "B1")
    rows["A3-All"] = next(row for row in deterministic_rows(summary) if row["policy"] == "A3-All")
    return rows


def save(fig, path, plt):
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def quality_frontier(summary, metric, output, plt):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    policies = ["R-Conf", "R-Margin", "R-Label", "R-Sim", "R-Static", "R-Memory", "R6"]
    for policy in policies:
        rows = sorted(
            (row for row in deterministic_rows(summary) if row["policy"] == policy),
            key=lambda row: row["route_rate"],
        )
        ax.plot(
            [row["avg_forward_passes"] for row in rows], [row[metric] for row in rows],
            marker="o", markersize=3, linewidth=1.2, color=COLORS[policy],
            label=DISPLAY[policy],
        )
    for policy, marker in (("B1", "s"), ("A3-All", "D")):
        row = next(row for row in deterministic_rows(summary) if row["policy"] == policy)
        ax.scatter([row["avg_forward_passes"]], [row[metric]], marker=marker, s=42,
                   color=COLORS[policy], label=DISPLAY[policy])
    ax.set_xlabel("Mean model forward passes per example")
    ax.set_ylabel("Accuracy" if metric == "accuracy" else "Macro-F1")
    ax.set_title(("Accuracy" if metric == "accuracy" else "Macro-F1") + " vs inference cost")
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.7)
    ax.legend(ncol=2, fontsize=7.5, loc="best")
    save(fig, output, plt)


def net_frontier(summary, output, plt):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    policies = ["R-Conf", "R-Margin", "R-Label", "R-Sim", "R-Static", "R-Memory", "R6"]
    for policy in policies:
        rows = sorted(
            (row for row in deterministic_rows(summary) if row["policy"] == policy),
            key=lambda row: row["route_rate"],
        )
        ax.plot([row["route_rate"] for row in rows], [row["net"] for row in rows],
                marker="o", markersize=3, linewidth=1.2, color=COLORS[policy],
                label=DISPLAY[policy])
    random = sorted(
        (row for row in summary["results"] if row["policy"] == "R-Random"),
        key=lambda row: row["requested_budget"],
    )
    ax.plot([row["aggregate"]["route_rate"]["mean"] for row in random],
            [row["aggregate"]["net"]["mean"] for row in random], linestyle="--",
            linewidth=1.2, color=COLORS["R-Random"], label="Random mean")
    ax.axhline(0, color="#6B7280", linewidth=0.8)
    ax.set_xlabel("Route rate")
    ax.set_ylabel("Helpful - harmful corrections")
    ax.set_title("Net correction gain vs route rate")
    ax.grid(axis="both", color="#E5E7EB", linewidth=0.7)
    ax.legend(ncol=2, fontsize=7.5, loc="best")
    save(fig, output, plt)


def helpful_harmful(summary, output, plt):
    rows = natural_rows(summary)
    policies = ["R-Conf", "R-Margin", "R-Label", "R-Sim", "R-Static", "R-Memory", "R6"]
    x = list(range(len(policies)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar([value - width / 2 for value in x], [rows[p]["helpful"] for p in policies],
           width, label="Helpful", color="#16A34A")
    ax.bar([value + width / 2 for value in x], [rows[p]["harmful"] for p in policies],
           width, label="Harmful", color="#DC2626")
    ax.set_xticks(x, [DISPLAY[p] for p in policies], rotation=28, ha="right")
    ax.set_ylabel("Number of examples")
    ax.set_title(f"Matched-budget flips at {summary['natural_r6_budget']:.1%} routing")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend()
    save(fig, output, plt)


def per_class_f1(summary, output, plt):
    rows = natural_rows(summary)
    strongest = summary["strongest_matched_budget_non_memory_router"]
    policies = ["B1", strongest, "R-Memory", "R6"]
    labels = sorted(rows["B1"]["per_class"])
    x = list(range(len(labels)))
    width = 0.19
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for index, policy in enumerate(policies):
        offset = (index - (len(policies) - 1) / 2) * width
        ax.bar([value + offset for value in x],
               [rows[policy]["per_class"][label]["f1"] for label in labels],
               width, label=DISPLAY[policy], color=COLORS[policy])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 at the natural R6 budget")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(ncol=2, fontsize=8)
    save(fig, output, plt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt = configure_plotting()
    quality_frontier(summary, "accuracy", args.output_dir / "accuracy_vs_forward_passes.pdf", plt)
    quality_frontier(summary, "macro_f1", args.output_dir / "macro_f1_vs_forward_passes.pdf", plt)
    net_frontier(summary, args.output_dir / "net_gain_vs_route_rate.pdf", plt)
    helpful_harmful(summary, args.output_dir / "helpful_harmful_by_router.pdf", plt)
    per_class_f1(summary, args.output_dir / "per_class_f1_comparison.pdf", plt)
    print(f"FIGURES_DONE count=5 output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
