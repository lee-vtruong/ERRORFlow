"""Audit downloaded FEVER JSONL files without changing them."""

import argparse
import json
from collections import Counter
from pathlib import Path


def audit(path: Path) -> dict:
    labels = Counter()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            labels[row.get("label", "MISSING")] += 1
    return {"file": str(path), "rows": rows, "labels": dict(labels)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    result = {"dataset": "FEVER", "version": "1.0", "splits": [audit(p) for p in args.files]}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
