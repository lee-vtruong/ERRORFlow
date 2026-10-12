"""Select the first k ranked evidence records for downstream verification."""
import argparse
import json
from pathlib import Path


def select_topk(row, k):
    evidence = row.get("evidence", [])
    metadata = row.get("evidence_metadata", [])
    if len(evidence) != len(metadata):
        raise ValueError(f"unaligned evidence and metadata for claim {row.get('claim_id')}")
    output = dict(row)
    output["evidence"] = evidence[:k]
    output["evidence_metadata"] = [dict(item, rank=rank) for rank, item in enumerate(metadata[:k], start=1)]
    output["retrieval"] = {
        "method": "top_k_selection",
        "source_retrieval": row.get("retrieval", {}),
        "k": k,
        "provenance": True,
    }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            target.write(json.dumps(select_topk(json.loads(line), args.k), ensure_ascii=False) + "\n")
            count += 1
    print(f"TOPK_DONE rows={count} k={args.k} output={args.output}")


if __name__ == "__main__":
    main()
