"""Filter canonical V2/V3 predictions to an exact example-ID subset."""

import argparse
from pathlib import Path

from errorflow.io import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ids = [str(row["example_id"]) for row in read_jsonl(args.examples)]
    if len(ids) != len(set(ids)):
        raise ValueError("Example subset contains duplicate IDs")
    predictions = {str(row["example_id"]): row for row in read_jsonl(args.predictions)}
    missing = set(ids) - set(predictions)
    if missing:
        raise ValueError(f"Predictions are missing {len(missing)} selected IDs")
    write_jsonl(args.output, (predictions[key] for key in ids))
    print(f"V3_FILTER_DONE rows={len(ids)} output={args.output}")


if __name__ == "__main__":
    main()
