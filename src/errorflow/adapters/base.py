"""Base helpers shared by dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

from errorflow.io import read_jsonl
from errorflow.schemas import NormalizedExample


class DatasetAdapter(ABC):
    name: str

    @abstractmethod
    def normalize(self, row: dict[str, Any], split: str) -> NormalizedExample:
        raise NotImplementedError

    def iter_file(self, path: str | Path, split: str) -> Iterator[NormalizedExample]:
        for row in read_jsonl(path):
            yield self.normalize(row, split)


def first_present(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default
