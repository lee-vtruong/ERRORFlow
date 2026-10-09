"""Measure recovery on a baseline error set."""
import argparse, json
from pathlib import Path

def load(path):
    return {str(json.loads(line)["claim_id"]): json.loads(line) for line in path.open(encoding="utf-8") if line.strip()}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--errors", type=Path, required=True); p.add_argument("--intervention", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    errors, intervention = load(a.errors), load(a.intervention); ids = sorted(set(errors) & set(intervention)); recovered = 0; still_wrong = 0; changed = 0
    for i in ids:
        gold = errors[i]["gold_label"]; before = errors[i]["baseline_prediction"]; after = intervention[i]["prediction"]
        recovered += after == gold; still_wrong += after != gold; changed += after != before
    result = {"n_baseline_errors": len(errors), "n_paired": len(ids), "missing_intervention_predictions": len(set(errors) - set(intervention)), "recovery_rate": recovered / len(ids) if ids else 0.0, "recovered": recovered, "still_wrong": still_wrong, "changed_prediction": changed, "intervention": next(iter(intervention.values()), {}).get("metadata", {}).get("intervention", "unknown")}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
