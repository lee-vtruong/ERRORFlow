"""Resolve FEVER evidence references against FEVER wiki-page JSONL records."""

import json
from pathlib import Path


def evidence_refs(record: dict) -> list[tuple[str, int]]:
    refs: list[tuple[str, int]] = []
    for group in record.get("evidence", []):
        for item in group:
            if len(item) < 4 or not item[2]:
                continue
            try:
                refs.append((str(item[2]), int(item[3])))
            except (TypeError, ValueError):
                continue
    return refs


def load_wiki_page(path: str | Path, wiki_id: str) -> dict | None:
    """Find one page in a FEVER wiki-pages JSONL shard."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and json.loads(line).get("id") == wiki_id:
                return json.loads(line)
    return None


def resolve_sentence(page: dict, sentence_id: int) -> str | None:
    for line in str(page.get("lines", "")).splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == sentence_id:
            return parts[1]
    return None
