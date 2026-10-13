"""Create paper-oriented external-validation PDF figures from frozen summaries."""

import argparse
import json
from pathlib import Path


POLICIES = ("B1", "A3-All", "R-Confidence", "R-Similarity", "R6")
FOCUS = ("B1", "A3-All", "R6")
DISPLAY = {
    "B1": "B1", "A3-All": "A3-All", "R-Confidence": "Confidence",
    "R-Similarity": "Similarity", "R6": "R6",
}
COLORS = {
    "B1": "#4C566A", "A3-All": "#9CA3AF", "R-Confidence": "#F59E0B",
    "R-Similarity": "#EC4899", "R6": "#DC2626",
}


def validate_summary(summary, name):
    missing = [policy for policy in POLICIES if policy not in summary.get("results", {})]
    if missing:
        raise ValueError(f"{name} summary missing policies: {missing}")
    return summary


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


def save(fig, path, plt):
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def quality_cost(summaries, metric, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=False)
    for ax, (name, summary) in zip(axes, summaries):
        for policy in POLICIES:
            row = summary["results"][policy]
            ax.scatter(
                row["avg_model_forward_passes"], row[metric], s=54,
                color=COLORS[policy], label=DISPLAY[policy], zorder=3,
            )
            ax.annotate(DISPLAY[policy],
                        (row["avg_model_forward_passes"], row[metric]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_title(name)
        ax.set_xlabel("Mean forward passes / example")
        ax.set_ylabel("Accuracy" if metric == "accuracy" else "Macro-F1")
        ax.grid(color="#E5E7EB", linewidth=0.7)
    save(fig, output, plt)


def per_class(summaries, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    width = 0.25
    for ax, (name, summary) in zip(axes, summaries):
        labels = list(summary["results"]["B1"]["per_class"])
        x = list(range(len(labels)))
        for index, policy in enumerate(FOCUS):
            offset = (index - 1) * width
            values = [summary["results"][policy]["per_class"][label]["f1"] for label in labels]
            ax.bar([value + offset for value in x], values, width,
                   color=COLORS[policy], label=DISPLAY[policy])
        ax.set_xticks(x, [label.replace("NOT ENOUGH INFO", "NEI") for label in labels],
                      rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(name)
        ax.set_ylabel("F1")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    axes[1].legend(loc="upper right")
    save(fig, output, plt)


def retrieval_groups(analysis, output, plt):
    order = ("nei_no_gold_rationale", "verifiable_hit", "verifiable_miss")
    names = ("NEI / no rationale", "Verifiable hit", "Verifiable miss")
    policies = ("A3-All", "R-Confidence", "R-Similarity", "R6")
    width = 0.19
    x = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for index, policy in enumerate(policies):
        offset = (index - 1.5) * width
        values = [analysis["groups"][group]["systems"][policy]["net_vs_b1"] for group in order]
        ax.bar([value + offset for value in x], values, width,
               color=COLORS[policy], label=DISPLAY[policy])
    ax.axhline(0, color="#6B7280", linewidth=0.8)
    ax.set_xticks(x, names)
    ax.set_ylabel("Helpful - harmful vs B1")
    ax.set_title("Correction utility by retrieval condition")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend(ncol=2)
    save(fig, output, plt)


def route_rates(analysis, output, plt):
    order = ("nei_no_gold_rationale", "verifiable_hit", "verifiable_miss")
    names = ("NEI / no rationale", "Verifiable hit", "Verifiable miss")
    policies = ("R-Confidence", "R-Similarity", "R6")
    width = 0.24
    x = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for index, policy in enumerate(policies):
        offset = (index - 1) * width
        values = [analysis["groups"][group]["systems"][policy]["route_rate"] for group in order]
        ax.bar([value + offset for value in x], values, width,
               color=COLORS[policy], label=DISPLAY[policy])
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.35)
    ax.set_ylabel("Route rate")
    ax.set_title("Selective routing by retrieval condition")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.legend()
    save(fig, output, plt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--climate-summary", required=True, type=Path)
    parser.add_argument("--scifact-summary", required=True, type=Path)
    parser.add_argument("--retrieval-analysis", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    climate = validate_summary(json.loads(args.climate_summary.read_text(encoding="utf-8")), "CLIMATE-FEVER")
    scifact = validate_summary(json.loads(args.scifact_summary.read_text(encoding="utf-8")), "SciFact retrieved")
    analysis = json.loads(args.retrieval_analysis.read_text(encoding="utf-8"))
    summaries = (("CLIMATE-FEVER", climate), ("SciFact retrieved", scifact))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt = configure_plotting()
    quality_cost(summaries, "accuracy", args.output_dir / "external_accuracy_vs_cost.pdf", plt)
    quality_cost(summaries, "macro_f1", args.output_dir / "external_macro_f1_vs_cost.pdf", plt)
    per_class(summaries, args.output_dir / "external_per_class_f1.pdf", plt)
    retrieval_groups(analysis, args.output_dir / "retrieval_condition_net_gain.pdf", plt)
    route_rates(analysis, args.output_dir / "retrieval_condition_route_rate.pdf", plt)
    print(f"EXTERNAL_FIGURES_DONE count=5 output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
