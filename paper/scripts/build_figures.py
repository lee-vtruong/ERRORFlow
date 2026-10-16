"""Build the vector figures used by the ERRORFlow paper."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "data" / "verified_results.json").read_text(encoding="utf-8"))
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
COLORS = {"B1": "#4C78A8", "A3-All": "#F58518", "R6": "#54A24B"}

def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()

def overview_method() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    def box(x, y, w, h, title, detail, color):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
            linewidth=1.1, edgecolor=color, facecolor=color + "18",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * .66, title, ha="center", va="center",
                fontsize=8.3, fontweight="bold", color="#222222")
        ax.text(x + w / 2, y + h * .30, detail, ha="center", va="center",
                fontsize=7, color="#333333", linespacing=1.25)

    def arrow(x1, y1, x2, y2, label=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=10, lw=1.0, color="#555555"))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + .18, label, ha="center",
                    fontsize=7, color="#555555")

    ax.text(.25, 4.72, "Offline: source outcome construction", fontsize=8,
            fontweight="bold", color="#555555")
    box(.25, 2.95, 2.15, 1.15, "Paired source runs",
        "Base + Alt\nhelpful / harmful / neutral", "#4C78A8")
    box(3.05, 2.95, 2.25, 1.15, "Outcome memory",
        "Claim, score profile, evidence size\nand observed effect", "#7A5195")
    arrow(2.40, 3.52, 3.05, 3.52)

    ax.plot([5.75, 5.75], [.35, 4.65], color="#BBBBBB", lw=.9, ls="--")
    ax.text(6.05, 4.72, "Online: selective prompt escalation", fontsize=8,
            fontweight="bold", color="#555555")
    box(6.05, 2.95, 1.65, 1.15, "Base prompt",
        "Verdict and\nlabel-score profile", "#4C78A8")
    box(8.25, 2.95, 1.80, 1.15, "Memory router",
        "Retrieve outcomes\nestimate signed utility", "#54A24B")
    box(10.70, 3.55, 1.15, .90, "Keep Base", "No extra\nmodel passes", "#4C78A8")
    box(10.70, 2.05, 1.15, .90, "Run Alt", "Fixed second\nprompt", "#F58518")
    arrow(7.70, 3.52, 8.25, 3.52)
    arrow(10.05, 3.65, 10.70, 4.00)
    arrow(10.05, 3.30, 10.70, 2.50)
    ax.text(10.39, 3.96, "$U\\leq0$", ha="center", fontsize=6.7, color="#555555")
    ax.text(10.39, 2.91, "$U>0$", ha="center", fontsize=6.7, color="#555555")
    ax.add_patch(FancyArrowPatch(
        (4.18, 2.95), (9.0, 2.95), arrowstyle="-|>", mutation_scale=10,
        lw=1.0, color="#555555", connectionstyle="arc3,rad=.22",
    ))
    ax.text(6.55, 2.12, "frozen source memory", ha="center", fontsize=7,
            color="#555555")
    ax.text(6.0, .38,
            "Non-parametric routing: source labels build memory; target labels are never used for routing.",
            ha="center", fontsize=7.5, color="#444444")
    save("overview_method.pdf")

def quality_cost() -> None:
    fig, axes = plt.subplots(1,2,figsize=(6.8,2.5))
    for ax, metric, title in zip(axes,["accuracy","macro_f1"],["Accuracy","Macro-F1"]):
        for ds,label in [("averitec","AV"),("climate_fever","CF"),("scifact_oracle_projected","SF-O"),("scifact_bm25_projected","SF-B")]:
            for sys in DATA[ds]["systems"]:
                x=DATA[ds]["systems"][sys]; ax.scatter(x["passes"],x[metric],color=COLORS[sys],marker={"AV":"o","CF":"s","SF-O":"^","SF-B":"D"}[label],s=35)
            r=DATA[ds]["systems"]["R6"]; ax.annotate(label,(r["passes"],r[metric]),xytext=(3,3),textcoords="offset points",fontsize=7)
        ax.set_xlabel("Average model forward passes"); ax.set_ylabel(title); ax.grid(alpha=.2)
    handles=[plt.Line2D([],[],marker='o',ls='',color=COLORS[k],label=k) for k in COLORS]
    fig.legend(handles=handles,loc="upper center",ncol=3,frameon=False); save("quality_cost.pdf")

def helpful_harmful() -> None:
    labels=[]; helpful=[]; harmful=[]
    for ds,short in [("averitec","AV"),("climate_fever","CF"),("scifact_bm25_projected","SF-B")]:
        for sys in ["A3-All","R6"]:
            x=DATA[ds]["systems"][sys]; labels.append(f"{short}\n{sys}"); helpful.append(x.get("helpful",0)); harmful.append(-x.get("harmful",0))
    fig,ax=plt.subplots(figsize=(6.8,2.5)); pos=range(len(labels)); ax.bar(pos,helpful,color="#54A24B",label="Helpful"); ax.bar(pos,harmful,color="#E45756",label="Harmful")
    ax.axhline(0,color="black",lw=.7); ax.set_xticks(list(pos),labels); ax.set_ylabel("Prediction changes"); ax.legend(frameon=False,ncol=2); save("helpful_harmful.pdf")

def retrieval_strata() -> None:
    st=DATA["scifact_bm25_projected"]["strata"]; names=list(st); x=range(len(names)); w=.24
    fig,ax=plt.subplots(figsize=(6.8,2.6))
    for j,sys in enumerate(["B1","A3-All","R6"]): ax.bar([i+(j-1)*w for i in x],[st[n][sys] for n in names],w,label=sys,color=COLORS[sys])
    ax.set_xticks(list(x),[f"{n}\n(n={st[n]['n']})" for n in names]); ax.set_ylabel("Accuracy"); ax.set_ylim(0,1); ax.legend(frameon=False,ncol=3); save("retrieval_strata.pdf")

def cross_domain_delta() -> None:
    ds=[("averitec","AVeriTeC"),("climate_fever","CLIMATE-FEVER"),("scifact_oracle_projected","SciFact oracle"),("scifact_bm25_projected","SciFact BM25")]
    acc=[DATA[k]["systems"]["R6"]["accuracy"]-DATA[k]["systems"]["B1"]["accuracy"] for k,_ in ds]
    f1=[DATA[k]["systems"]["R6"]["macro_f1"]-DATA[k]["systems"]["B1"]["macro_f1"] for k,_ in ds]
    acc_ci=[DATA[k]["r6_ci95"]["accuracy"] for k,_ in ds]
    f1_ci=[DATA[k]["r6_ci95"]["macro_f1"] for k,_ in ds]
    fig,ax=plt.subplots(figsize=(6.8,3.0)); y=list(range(len(ds))); offset=.11
    for values,cis,dy,label,color,marker in [
        (acc,acc_ci,-offset,"Accuracy","#4C78A8","o"),
        (f1,f1_ci,offset,"Macro-F1","#F58518","s"),
    ]:
        lower=[v-ci[0] for v,ci in zip(values,cis)]
        upper=[ci[1]-v for v,ci in zip(values,cis)]
        ax.errorbar(values,[i+dy for i in y],xerr=[lower,upper],fmt=marker,
                    color=color,capsize=3,label=label,lw=1.2,markersize=5)
    ax.axvline(0,color="black",lw=.8,ls="--")
    ax.set_yticks(y,[v for _,v in ds]); ax.invert_yaxis()
    ax.set_xlabel("R6 minus B1 (paired 95% bootstrap CI)")
    ax.annotate("adapter-sensitive",(f1[2],2+offset),xytext=(8,7),
                textcoords="offset points",fontsize=7)
    ax.legend(frameon=False,ncol=2,loc="lower right"); ax.grid(axis="x",alpha=.2)
    save("cross_domain_delta.pdf")

if __name__ == "__main__":
    overview_method(); quality_cost(); helpful_harmful(); retrieval_strata(); cross_domain_delta()
    print(f"FIGURES_DONE count=5 output_dir={OUT}")
