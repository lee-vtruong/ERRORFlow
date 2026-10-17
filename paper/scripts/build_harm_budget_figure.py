"""Build Figure 2 with a random-routing containment reference."""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "helpful_harmful.pdf"


def main():
    plt.rcParams.update({
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))

    ax = axes[0]
    settings = [
        ("Qwen AV", 18.54, 48.96, (5, -12)),
        ("Mistral AV", 19.57, 55.81, (-2, 8)),
        ("Climate in-domain", 39.51, 52.34, (5, 4)),
        ("SciFact BM25\n(audit prompts)", 19.35, 20.0, (5, 4)),
    ]
    diagonal = list(range(0, 101))
    ax.plot(diagonal, diagonal, "--", color="#777777", linewidth=1.1)
    for index, (name, incurred, retained, offset) in enumerate(settings):
        ax.scatter(incurred, retained, s=46, zorder=4,
                   color=["#4C78A8", "#7A5195", "#54A24B", "#F58518"][index])
        ax.annotate(name, (incurred, retained), xytext=offset,
                    textcoords="offset points", fontsize=7)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 63)
    ax.set_xlabel("Harmful Audit-Alt changes incurred (%)")
    ax.set_ylabel("Helpful Audit-Alt changes retained (%)")
    ax.set_title("(a) Enrichment relative to random routing")
    ax.grid(alpha=.18)

    ax = axes[1]
    budgets = [0, 5, 10, 11.18, 15, 20, 25, 30, 35, 40]
    full = [74.02, 74.93, 75.52, 75.75, 75.88, 75.88, 75.85, 75.55, 75.62, 75.55]
    error_only = [74.02, 75.16, 75.55, 75.65, 75.88, 75.68, 75.33, 75.07, 74.87, 74.61]
    recovery = [74.02, 75.23, 75.75, 75.78, 76.14, 75.95, 75.81, 75.52, 75.33, 75.16]
    ax.plot(budgets, full, "o-", color="#2F7F3F", label="Full outcome memory")
    ax.plot(budgets, recovery, "s--", color="#E17C24", label="Recovery-only utility")
    ax.plot(budgets, error_only, "^:", color="#7651A8", label="Error-only memory")
    ax.axhline(74.02, color="#666666", linestyle="--", linewidth=.9,
               label="Audit-Base (4 passes)")
    ax.axhline(76.86, color="#1F77B4", linestyle="-.", linewidth=.9,
               label="Standard-Base (4 passes)")
    ax.axhline(75.62, color="#A020C0", linestyle=(0, (6, 3)), linewidth=.9,
               label="Score ensemble (8 passes)")
    ax.axhline(73.60, color="#D62728", linestyle=":", linewidth=.9,
               label="Audit-Alt-All (8 passes)")
    ax.scatter([12.22], [75.85], marker="*", s=130, color="#F2B134",
               edgecolor="#222222", zorder=5)
    ax.annotate("Natural\n(12.22%, 75.85)", (12.22, 75.85), xytext=(18, -50),
                textcoords="offset points", fontsize=7,
                arrowprops={"arrowstyle": "-", "color": "#444444", "lw": .8})
    ax.set_xlabel("Routed to Audit-Alt (% of claims)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(73.45, 77.05)
    ax.set_title("(b) AVeriTeC accuracy vs. correction budget")
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper center",
              bbox_to_anchor=(.5, -.25))
    top = ax.secondary_xaxis("top", functions=(lambda value: 4 + .04 * value, lambda value: (value - 4) / .04))
    top.set_xlabel("Average forward passes = $4+4r$")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"FIGURE_DONE output={OUT}")


if __name__ == "__main__":
    main()
