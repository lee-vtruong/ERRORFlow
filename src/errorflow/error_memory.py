"""Structured V2 error-memory store with deterministic lexical retrieval."""

from __future__ import annotations

import hashlib
import math
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


def score_distribution(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    maximum = max(scores.values())
    exponentials = {label: math.exp(value - maximum) for label, value in scores.items()}
    total = sum(exponentials.values())
    return {label: value / total for label, value in exponentials.items()}


def distribution_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    a, b = score_distribution(left), score_distribution(right)
    if not a or not b:
        return 0.0
    labels = set(a) | set(b)
    return 1.0 - 0.5 * sum(abs(a.get(label, 0.0) - b.get(label, 0.0)) for label in labels)


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
        allowed_splits: set[str] | None = None, baseline_prediction: str | None = None,
        label_scores: dict[str, float] | None = None, exclude_claim_ids: set[str] | None = None,
        action: str | None = None,
    ) -> list[tuple[ErrorMemoryRecord, float]]:
        scored = []
        for record in self._records.values():
            if not record.allow_for_routing:
                continue
            if dataset and record.dataset != dataset:
                continue
            if allowed_splits is not None and record.source_split not in allowed_splits:
                continue
            if baseline_prediction and record.baseline_prediction != baseline_prediction:
                continue
            if exclude_claim_ids and record.claim_id in exclude_claim_ids:
                continue
            if action and record.action != action:
                continue
            lexical = jaccard_similarity(claim, record.claim)
            historical_scores = record.metadata.get("baseline_label_scores", {})
            profile = distribution_similarity(label_scores or {}, historical_scores)
            score = 0.4 * lexical + 0.6 * profile if label_scores and historical_scores else lexical
            if score > 0:
                scored.append((record, score))
        return sorted(scored, key=lambda item: (-item[1], item[0].memory_id))[:top_k]

    def __len__(self) -> int:
        return len(self._records)
