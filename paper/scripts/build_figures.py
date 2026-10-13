"""Build the vector figures used by the ERRORFlow paper."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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

def architecture() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 2.15)); ax.axis("off")
    boxes = [(0.02, .37, .16, .28, "Claim + evidence"), (.25, .37, .15, .28, "B1 frozen\nlabel scorer"),
             (.48, .56, .18, .28, "Residual memory\n(source outcomes)"), (.48, .16, .18, .28, "R6 utility\nrouter"),
             (.75, .16, .16, .28, "A3 frozen\ncorrection")]
    for x,y,w,h,s in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.015",fc="#F2F2F2",ec="#333333")); ax.text(x+w/2,y+h/2,s,ha="center",va="center")
    arrows=[((.18,.51),(.25,.51)),((.40,.51),(.48,.30)),((.66,.30),(.75,.30)),((.66,.70),(.57,.44)),((.91,.30),(.97,.30)),((.40,.51),(.97,.70))]
    for a,b in arrows: ax.add_patch(FancyArrowPatch(a,b,arrowstyle="->",mutation_scale=10,color="#333333"))
    ax.text(.93,.73,"keep B1 if not routed",ha="center",fontsize=7); ax.text(.94,.22,"final label",ha="center",fontsize=7)
    ax.set_xlim(0,1); ax.set_ylim(0,1); save("architecture.pdf")

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
    fig,ax=plt.subplots(figsize=(6.8,2.6)); x=range(len(ds)); w=.34
    ax.bar([i-w/2 for i in x],acc,w,label="Accuracy",color="#4C78A8"); ax.bar([i+w/2 for i in x],f1,w,label="Macro-F1",color="#F58518")
    ax.axhline(0,color="black",lw=.7); ax.set_xticks(list(x),[v for _,v in ds]); ax.set_ylabel("R6 minus B1"); ax.legend(frameon=False,ncol=2); save("cross_domain_delta.pdf")

if __name__ == "__main__":
    architecture(); quality_cost(); helpful_harmful(); retrieval_strata(); cross_domain_delta()
    print(f"FIGURES_DONE count=5 output_dir={OUT}")
