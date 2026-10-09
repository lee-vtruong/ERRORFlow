"""Create conservative, observable error diagnoses and candidate actions."""

import argparse, json
from pathlib import Path

def diagnose(row):
    gold, pred = row["gold_label"], row["baseline_prediction"]
    if pred == "NOT ENOUGH INFO":
        basis, actions = "verifier returned insufficiency despite retrieved context", ["evidence_critic", "reverify"]
    elif gold == "NOT ENOUGH INFO":
        basis, actions = "verifier selected a verdict where annotation is NEI", ["evidence_critic", "conflict_check"]
    else:
        basis, actions = "supported/refuted polarity confusion", ["contradiction_critic", "reverify"]
    return {**row, "diagnosis_basis": basis, "candidate_actions": actions, "diagnosis_status": "heuristic_candidate"}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--input", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args(); a.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with a.input.open(encoding="utf-8") as source, a.output.open("w", encoding="utf-8") as target:
        for line in source:
            if line.strip(): target.write(json.dumps(diagnose(json.loads(line)), ensure_ascii=False) + "\n"); n += 1
    print(f"diagnosed={n} output={a.output}")

if __name__ == "__main__": main()
