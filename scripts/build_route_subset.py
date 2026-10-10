"""Select claims by observable baseline prediction for route evaluation."""
import argparse, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--claims",type=Path,required=True); p.add_argument("--baseline",type=Path,required=True); p.add_argument("--prediction",required=True); p.add_argument("--max-confidence",type=float); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    pred={str(json.loads(x)["claim_id"]):json.loads(x) for x in a.baseline.open(encoding="utf-8") if x.strip()}; n=0; a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.claims.open(encoding="utf-8") as src,a.output.open("w",encoding="utf-8") as out:
        for line in src:
            if line.strip():
                row=json.loads(line)
                baseline=pred.get(str(row["claim_id"]),{})
                confidence=baseline.get("confidence")
                confidence_ok=a.max_confidence is None or (confidence is not None and confidence<=a.max_confidence)
                if baseline.get("prediction")==a.prediction and confidence_ok: out.write(json.dumps(row,ensure_ascii=False)+"\n"); n+=1
    print(f"route_records={n} prediction={a.prediction} max_confidence={a.max_confidence} output={a.output}")
if __name__=="__main__": main()
