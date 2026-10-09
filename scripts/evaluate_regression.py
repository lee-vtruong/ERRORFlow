"""Evaluate intervention recovery and regression against full baseline."""
import argparse, json
from pathlib import Path

def load(path): return {str(json.loads(x)["claim_id"]): json.loads(x) for x in path.open(encoding="utf-8") if x.strip()}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--gold", type=Path, required=True); p.add_argument("--baseline", type=Path, required=True); p.add_argument("--intervention", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    gold, base, inter = load(a.gold), load(a.baseline), load(a.intervention); ids = sorted(set(gold) & set(base) & set(inter)); recovered = regressed = base_correct = base_wrong = 0
    for i in ids:
        g, b, x = gold[i]["gold_label"], base[i]["prediction"], inter[i]["prediction"]
        if b == g: base_correct += 1; regressed += x != g
        else: base_wrong += 1; recovered += x == g
    result = {"n_paired": len(ids), "baseline_correct": base_correct, "baseline_wrong": base_wrong, "recovered": recovered, "recovery_rate": recovered / base_wrong if base_wrong else 0.0, "regressed": regressed, "regression_rate": regressed / base_correct if base_correct else 0.0, "net_correct_change": recovered - regressed, "intervention": next(iter(inter.values()), {}).get("metadata", {}).get("intervention", "unknown")}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
