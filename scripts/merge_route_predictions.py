"""Apply intervention predictions only to a routed subset."""
import argparse, json
from pathlib import Path

def load(path): return {str(json.loads(x)["claim_id"]): json.loads(x) for x in path.open(encoding="utf-8") if x.strip()}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--baseline",type=Path,required=True); p.add_argument("--intervention",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    base, inter=load(a.baseline),load(a.intervention); a.output.parent.mkdir(parents=True,exist_ok=True); changed=0
    with a.output.open("w",encoding="utf-8") as out:
        for claim_id,row in base.items():
            if claim_id in inter:
                row=dict(inter[claim_id]); row["metadata"]={**row.get("metadata",{}),"route_applied":True}; changed+=1
            else:
                row=dict(row); row["metadata"]={**row.get("metadata",{}),"route_applied":False}
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(f"merged={len(base)} route_applied={changed} output={a.output}")
if __name__=="__main__": main()
