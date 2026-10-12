"""Structured V2 error-memory store with deterministic lexical retrieval."""

from __future__ import annotations

import hashlib
import re
from dataclasses import fields
from pathlib import Path
from typing import Iterable

from .io import read_jsonl, write_jsonl
from .schemas import ErrorMemoryRecord

TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


def memory_id(dataset: str, claim_id: str, error_type: str, action: str) -> str:
    raw = "\x1f".join((dataset, claim_id, error_type, action))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def jaccard_similarity(left: str, right: str) -> float:
    a, b = tokens(left), tokens(right)
    return len(a & b) / len(a | b) if a or b else 0.0


def record_from_dict(payload: dict) -> ErrorMemoryRecord:
    allowed = {item.name for item in fields(ErrorMemoryRecord)}
    return ErrorMemoryRecord(**{key: value for key, value in payload.items() if key in allowed})


class ErrorMemoryStore:
    def __init__(self, records: Iterable[ErrorMemoryRecord] = ()):
        self._records: dict[str, ErrorMemoryRecord] = {}
        for record in records:
            self.upsert(record)

    @classmethod
    def load(cls, path: str | Path) -> "ErrorMemoryStore":
        return cls(record_from_dict(row) for row in read_jsonl(path))

    def upsert(self, record: ErrorMemoryRecord) -> None:
        self._records[record.memory_id] = record

    def save(self, path: str | Path) -> None:
        write_jsonl(path, (self._records[key].to_dict() for key in sorted(self._records)))

    def retrieve(
        self, claim: str, *, dataset: str | None = None, top_k: int = 5,
        allowed_splits: set[str] | None = None,
    ) -> list[tuple[ErrorMemoryRecord, float]]:
        scored = []
        for record in self._records.values():
            if not record.allow_for_routing:
                continue
            if dataset and record.dataset != dataset:
                continue
            if allowed_splits is not None and record.source_split not in allowed_splits:
                continue
            score = jaccard_similarity(claim, record.claim)
            if score > 0:
                scored.append((record, score))
        return sorted(scored, key=lambda item: (-item[1], item[0].memory_id))[:top_k]

    def __len__(self) -> int:
        return len(self._records)
