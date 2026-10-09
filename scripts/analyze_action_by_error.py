"""Summarize intervention recovery by observable error type."""
import argparse, json
from collections import defaultdict
from pathlib import Path

def load(path): return {str(json.loads(x)["claim_id"]): json.loads(x) for x in path.open(encoding="utf-8") if x.strip()}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--errors", type=Path, required=True); p.add_argument("--intervention", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    errors, action = load(a.errors), load(a.intervention); groups = defaultdict(lambda: {"n": 0, "recovered": 0, "changed": 0})
    for claim_id, row in errors.items():
        if claim_id not in action: continue
        group = groups[row["error_type_observable"]]; group["n"] += 1
        group["recovered"] += action[claim_id]["prediction"] == row["gold_label"]
        group["changed"] += action[claim_id]["prediction"] != row["baseline_prediction"]
    result = {k: {**v, "recovery_rate": v["recovered"] / v["n"] if v["n"] else 0.0} for k, v in sorted(groups.items())}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
