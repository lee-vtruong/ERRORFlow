"""Sweep confidence thresholds for a selective intervention route on train/dev."""
import argparse, json
from collections import Counter
from pathlib import Path

def load(path): return {str(json.loads(x)["claim_id"]):json.loads(x) for x in path.open(encoding="utf-8") if x.strip()}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--gold",type=Path,required=True); p.add_argument("--baseline",type=Path,required=True); p.add_argument("--intervention",type=Path,required=True); p.add_argument("--prediction",default="REFUTED"); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    gold,base,inter=load(a.gold),load(a.baseline),load(a.intervention); thresholds=[i/20 for i in range(1,21)]; rows=[]
    for threshold in thresholds:
        correct=routed=0; confusion=Counter(); labels=("SUPPORTED","REFUTED","NOT ENOUGH INFO")
        for claim_id,g in gold.items():
            b=base[claim_id]; use=b["prediction"]==a.prediction and b.get("confidence") is not None and b["confidence"]<=threshold and claim_id in inter
            prediction=inter[claim_id]["prediction"] if use else b["prediction"]; routed+=use; correct+=prediction==g["gold_label"]; confusion[(g["gold_label"],prediction)]+=1
        f1s=[]
        for label in labels:
            tp=confusion[(label,label)]; fp=sum(confusion[(other,label)] for other in labels if other!=label); fn=sum(confusion[(label,other)] for other in labels if other!=label); f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
        rows.append({"max_confidence":threshold,"routed":routed,"accuracy":correct/len(gold),"macro_f1":sum(f1s)/len(f1s)})
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8"); print(json.dumps(rows,indent=2))
if __name__=="__main__": main()
