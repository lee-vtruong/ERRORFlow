"""SciFact claim adapter supporting the official JSONL field shapes."""

from __future__ import annotations

from typing import Any

from errorflow.schemas import EvidenceItem, NormalizedExample

from .base import DatasetAdapter, first_present


SCIFACT_LABELS = {"SUPPORT": "SUPPORTED", "CONTRADICT": "REFUTED", "NEI": "NOT ENOUGH INFO"}


class SciFactAdapter(DatasetAdapter):
    name = "SciFact"

    def normalize(self, row: dict[str, Any], split: str) -> NormalizedExample:
        raw_label = first_present(row, "label", "gold_label")
        label = SCIFACT_LABELS.get(str(raw_label).upper(), raw_label) if raw_label else None
        evidence = []
        for index, item in enumerate(first_present(row, "evidence_text", "evidence", default=[])):
            if isinstance(item, str):
                evidence.append(EvidenceItem(str(index), item, source="scientific_abstract"))
            elif isinstance(item, dict) and item.get("text"):
                evidence.append(EvidenceItem(str(item.get("id", index)), item["text"], title=item.get("title"), source="scientific_abstract"))
        claim_id = first_present(row, "id", "claim_id")
        return NormalizedExample(str(claim_id), self.name, split, row["claim"], label, evidence, {"cited_doc_ids": row.get("cited_doc_ids", [])})
