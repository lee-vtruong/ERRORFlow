"""Merge title-derived candidates ahead of lexical candidates without duplicates."""
import argparse
import json
from pathlib import Path


def merge_row(lexical, title, k=50, title_first=10):
    lexical_pairs = list(zip(lexical.get("evidence", []), lexical.get("evidence_metadata", [])))
    title_pairs = list(zip(title.get("evidence", []), title.get("evidence_metadata", [])))
    ordered = title_pairs[:title_first] + lexical_pairs + title_pairs[title_first:]
    merged = []
    seen = set()
    for text, metadata in ordered:
        key = (str(metadata["page_id"]), int(metadata["sentence_id"]))
        if key in seen:
            continue
        seen.add(key)
        merged.append((text, dict(metadata)))
        if len(merged) == k:
            break
    output = dict(lexical)
    output["evidence"] = [text for text, _ in merged]
    output["evidence_metadata"] = []
    for rank, (_, metadata) in enumerate(merged, start=1):
        metadata["rank"] = rank
        metadata.setdefault("source", "lexical_bm25")
        output["evidence_metadata"].append(metadata)
    output["retrieval"] = {
        "method": "title_first_then_lexical",
        "title_first": title_first,
        "k": k,
        "provenance": True,
    }
    return output


def read_by_id(path):
    with path.open(encoding="utf-8") as source:
        return {str(row["claim_id"]): row for row in map(json.loads, filter(str.strip, source))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexical", type=Path, required=True)
    parser.add_argument("--title", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--title-first", type=int, default=10)
    args = parser.parse_args()
    lexical = read_by_id(args.lexical)
    title = read_by_id(args.title)
    if set(lexical) != set(title):
        raise ValueError("lexical and title artifacts must contain the same claim IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        for claim_id, row in lexical.items():
            target.write(json.dumps(merge_row(row, title[claim_id], args.k, args.title_first), ensure_ascii=False) + "\n")
    print(f"MERGE_DONE rows={len(lexical)} output={args.output}")


if __name__ == "__main__":
    main()
