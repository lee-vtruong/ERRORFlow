"""Filter a prediction artifact to an explicit claim subset."""
import argparse
import json
from pathlib import Path


def load_rows(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def select_predictions(claim_rows, prediction_rows):
    predictions = {str(row["claim_id"]): row for row in prediction_rows}
    selected = []
    missing = []
    seen = set()
    for claim in claim_rows:
        claim_id = str(claim["claim_id"])
        if claim_id in seen:
            raise ValueError(f"duplicate claim_id in claims: {claim_id}")
        seen.add(claim_id)
        if claim_id not in predictions:
            missing.append(claim_id)
        else:
            selected.append(predictions[claim_id])
    if missing:
        raise ValueError(f"missing {len(missing)} predictions for selected claims")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = select_predictions(load_rows(args.claims), load_rows(args.predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for row in selected:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"FILTER_DONE rows={len(selected)} output={args.output}")


if __name__ == "__main__":
    main()
