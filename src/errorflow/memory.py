"""Persistent JSONL error memory."""

import json
from pathlib import Path
from typing import Iterable

from .schemas import ErrorMemoryEntry


def save_jsonl(entries: Iterable[ErrorMemoryEntry], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[ErrorMemoryEntry]:
    entries: list[ErrorMemoryEntry] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                entries.append(ErrorMemoryEntry(**payload))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid error-memory record at line {line_number}") from exc
    return entries
