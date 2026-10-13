"""Apply a predeclared label-space projection to cached predictions."""

import argparse
from pathlib import Path

from errorflow.io import read_jsonl, write_jsonl


def project(rows, mapping):
    for source in rows:
        row = dict(source)
        original = row["prediction"]
        row["prediction"] = mapping.get(original, original)
        row["metadata"] = dict(row.get("metadata", {}))
        row["metadata"].update({
            "pre_projection_prediction": original,
            "label_projection": mapping,
        })
        yield row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conflicting-to", choices=["NOT ENOUGH INFO"], default="NOT ENOUGH INFO")
    args = parser.parse_args()
    mapping = {"CONFLICTING": args.conflicting_to}
    rows = list(project(read_jsonl(args.input), mapping))
    write_jsonl(args.output, rows)
    changed = sum(row["metadata"]["pre_projection_prediction"] != row["prediction"] for row in rows)
    print(f"V3_LABEL_PROJECTION_DONE rows={len(rows)} changed={changed} output={args.output}")


if __name__ == "__main__":
    main()
