"""Build counterfactual intervention outcomes from train-side runs."""
import argparse, json
from pathlib import Path

def load(path):
    return {str(json.loads(line)["claim_id"]): json.loads(line) for line in path.open(encoding="utf-8") if line.strip()}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--errors", type=Path, required=True); p.add_argument("--intervention", action="append", nargs=2, metavar=("NAME", "PATH"), required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    errors = load(a.errors); runs = {name: load(Path(path)) for name, path in a.intervention}; a.output.parent.mkdir(parents=True, exist_ok=True); n = 0
    with a.output.open("w", encoding="utf-8") as target:
        for claim_id, error in errors.items():
            outcomes = {}; best = None
            for name, records in runs.items():
                prediction = records[claim_id]["prediction"]; success = prediction == error["gold_label"]; outcomes[name] = {"prediction": prediction, "recovery_success": success}
                if success and best is None: best = name
            row = {**error, "candidate_actions": list(runs), "intervention_outcomes": outcomes, "best_action": best, "memory_status": "counterfactual_train_observation"}
            target.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
    print(f"memory_records={n} output={a.output}")

if __name__ == "__main__": main()
