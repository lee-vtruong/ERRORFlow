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
                baseline_row = row
                row=dict(inter[claim_id])
                row["token_count"] = int(baseline_row.get("token_count", 0)) + int(row.get("token_count", 0))
                row["latency_ms"] = float(baseline_row.get("latency_ms", 0.0)) + float(row.get("latency_ms", 0.0))
                row["metadata"]={
                    **row.get("metadata",{}),
                    "route_applied":True,
                    "llm_calls":2,
                    "baseline_prediction":baseline_row.get("prediction"),
                    "baseline_token_count":baseline_row.get("token_count",0),
                    "intervention_token_count":inter[claim_id].get("token_count",0),
                }
                changed+=1
            else:
                row=dict(row); row["metadata"]={**row.get("metadata",{}),"route_applied":False,"llm_calls":1}
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(f"merged={len(base)} route_applied={changed} output={a.output}")
if __name__=="__main__": main()
