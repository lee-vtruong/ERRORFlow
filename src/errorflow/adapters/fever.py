"""FEVER 1.0 adapter."""

from __future__ import annotations

from typing import Any

from errorflow.datasets import normalize_fever_label
from errorflow.schemas import EvidenceItem, NormalizedExample

from .base import DatasetAdapter


class FeverAdapter(DatasetAdapter):
    name = "FEVER"

    def normalize(self, row: dict[str, Any], split: str) -> NormalizedExample:
        raw_label = row.get("label") or row.get("gold_label")
        aliases = {"SUPPORTED": "SUPPORTED", "REFUTED": "REFUTED", "NOT ENOUGH INFO": "NOT ENOUGH INFO"}
        label = aliases.get(str(raw_label).upper()) if raw_label else None
        if raw_label and label is None:
            label = normalize_fever_label(str(raw_label).upper())
        evidence = []
        for index, item in enumerate(row.get("evidence", [])):
            if isinstance(item, str):
                evidence.append(EvidenceItem(str(index), item, source="provided_text"))
            elif isinstance(item, dict) and item.get("text"):
                evidence.append(EvidenceItem(
                    str(item.get("evidence_id", index)), item["text"],
                    title=item.get("title"), source=item.get("source"), metadata=item.get("metadata", {}),
                ))
        return NormalizedExample(
            example_id=str(row.get("id", row.get("claim_id"))), dataset=self.name,
            split=split, claim=row["claim"], label=label, evidence=evidence,
            metadata={"raw_evidence": row.get("evidence", [])} if not evidence else {},
        )
