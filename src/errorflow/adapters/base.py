"""Base helpers shared by dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
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
        source = Path(path)
        with source.open(encoding="utf-8") as handle:
            first = handle.read(1)
            handle.seek(0)
            if first == "[":
                rows = json.load(handle)
                if not isinstance(rows, list):
                    raise ValueError(f"Expected a JSON array in {source}")
            else:
                rows = read_jsonl(source)
        for index, row in enumerate(rows):
            indexed = dict(row)
            indexed.setdefault("_row_index", index)
            yield self.normalize(indexed, split)


def first_present(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default
