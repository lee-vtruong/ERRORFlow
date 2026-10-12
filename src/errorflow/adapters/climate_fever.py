"""Climate-FEVER adapter."""

from __future__ import annotations

from typing import Any

from errorflow.datasets import normalize_fever_label
from errorflow.schemas import EvidenceItem, NormalizedExample

from .base import DatasetAdapter, first_present


class ClimateFeverAdapter(DatasetAdapter):
    name = "Climate-FEVER"

    def normalize(self, row: dict[str, Any], split: str) -> NormalizedExample:
        raw_label = first_present(row, "claim_label", "label", "gold_label")
        aliases = {
            "SUPPORTS": "SUPPORTED", "SUPPORTED": "SUPPORTED",
            "REFUTES": "REFUTED", "REFUTED": "REFUTED",
            "NOT_ENOUGH_INFO": "NOT ENOUGH INFO", "NOT ENOUGH INFO": "NOT ENOUGH INFO",
            "DISPUTED": "CONFLICTING",
        }
        label = aliases.get(str(raw_label).upper()) if raw_label else None
        if raw_label and label is None:
            label = normalize_fever_label(str(raw_label).upper())
        evidence = []
        for index, item in enumerate(row.get("evidences", row.get("evidence", []))):
            if isinstance(item, str):
                evidence.append(EvidenceItem(str(index), item, source="wikipedia"))
            elif isinstance(item, dict):
                text = first_present(item, "evidence", "text", default="")
                if text:
                    evidence.append(EvidenceItem(str(item.get("evidence_id", index)), text, title=item.get("article"), source="wikipedia"))
        claim_id = first_present(row, "claim_id", "id")
        return NormalizedExample(str(claim_id), self.name, split, row["claim"], label, evidence)
