"""Build an auditable error list from prepared gold rows and predictions."""

import argparse
import json
from pathlib import Path


def observable_type(gold: str, prediction: str, evidence: list[str]) -> str:
    if not evidence and gold == "NOT ENOUGH INFO":
        return "nei_no_evidence"
    if not evidence:
        return "missing_evidence_context"
    return f"{gold.lower().replace(' ', '_')}_predicted_{prediction.lower().replace(' ', '_')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gold = {str(json.loads(line)["claim_id"]): json.loads(line) for line in args.gold.open(encoding="utf-8") if line.strip()}
    pred = {str(json.loads(line)["claim_id"]): json.loads(line) for line in args.predictions.open(encoding="utf-8") if line.strip()}
    errors = []
    for claim_id, row in gold.items():
        prediction = pred.get(claim_id)
        if prediction is None or prediction["prediction"] == row["gold_label"]:
            continue
        evidence = row.get("evidence", [])
        errors.append({
            "claim_id": claim_id,
            "claim": row["claim"],
            "gold_label": row["gold_label"],
            "baseline_prediction": prediction["prediction"],
            "error_type_observable": observable_type(row["gold_label"], prediction["prediction"], evidence),
            "evidence_count": len(evidence),
            "raw_output": prediction.get("metadata", {}).get("raw_output", ""),
            "model": prediction.get("model"),
            "split": prediction.get("split"),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    print(f"errors={len(errors)} output={args.output}")


if __name__ == "__main__":
    main()
