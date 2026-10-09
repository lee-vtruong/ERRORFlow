"""Create a baseline-correct control subset for regression testing."""
import argparse, json
from pathlib import Path

def load(path): return {str(json.loads(x)["claim_id"]): json.loads(x) for x in path.open(encoding="utf-8") if x.strip()}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--claims", type=Path, required=True); p.add_argument("--baseline", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    claims, baseline = load(a.claims), load(a.baseline); a.output.parent.mkdir(parents=True, exist_ok=True); n = 0
    with a.output.open("w", encoding="utf-8") as out:
        for claim_id, row in claims.items():
            if claim_id in baseline and baseline[claim_id]["prediction"] == row["gold_label"]:
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
    print(f"control_records={n} output={a.output}")

if __name__ == "__main__": main()
