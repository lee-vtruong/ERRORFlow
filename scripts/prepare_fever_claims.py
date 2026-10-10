"""Prepare FEVER claims and labels without consulting annotated evidence."""
import argparse
import json
from pathlib import Path

from errorflow.datasets import normalize_fever_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            if args.limit is not None and written >= args.limit:
                break
            row = json.loads(line)
            target.write(json.dumps({
                "claim_id": str(row["id"]),
                "claim": row["claim"],
                "evidence": [],
                "gold_label": normalize_fever_label(row["label"]),
                "dataset": "FEVER",
                "dataset_version": "1.0",
                "source_split": args.input.stem,
                "preparation": "claim_only_no_gold_evidence",
            }, ensure_ascii=False) + "\n")
            written += 1
    print(f"prepared={written} output={args.output}")


if __name__ == "__main__":
    main()
