"""Prepare a small FEVER subset with resolved evidence passages."""

import argparse
import json
from pathlib import Path

from errorflow.datasets import normalize_fever_label
from errorflow.fever_evidence import evidence_refs, resolve_sentence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--wiki-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    selected = []
    with args.claims.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and len(selected) < args.limit:
                row = json.loads(line)
                selected.append(row)

    needed = {page_id for row in selected for page_id, _ in evidence_refs(row)}
    pages: dict[str, dict] = {}
    for shard in sorted(args.wiki_dir.glob("*.jsonl")):
        if not needed:
            break
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                page = json.loads(line)
                if page.get("id") in needed:
                    pages[page["id"]] = page
                    needed.remove(page["id"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = 0
    with args.output.open("w", encoding="utf-8") as target:
        for row in selected:
            evidence = []
            for page_id, sentence_id in evidence_refs(row):
                page = pages.get(page_id)
                if page:
                    sentence = resolve_sentence(page, sentence_id)
                    if sentence:
                        evidence.append(sentence)
            if row.get("label") != "NOT ENOUGH INFO" and not evidence:
                missing += 1
                continue
            target.write(json.dumps({
                "claim_id": str(row["id"]),
                "claim": row["claim"],
                "evidence": evidence,
                "gold_label": normalize_fever_label(row["label"]),
                "dataset": "FEVER",
                "dataset_version": "1.0",
                "source_split": args.claims.stem,
            }, ensure_ascii=False) + "\n")
            written += 1
    print(f"selected={len(selected)} written={written} missing_positive_evidence={missing}")


if __name__ == "__main__":
    main()
