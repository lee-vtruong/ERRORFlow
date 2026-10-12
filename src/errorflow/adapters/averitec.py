"""AVeriTeC adapter for claim/question-answer evidence records."""

from __future__ import annotations

from typing import Any

from errorflow.schemas import EvidenceItem, NormalizedExample

from .base import DatasetAdapter, first_present


AVERITEC_LABELS = {
    "SUPPORTED": "SUPPORTED", "REFUTED": "REFUTED",
    "NOT_ENOUGH_EVIDENCE": "NOT ENOUGH INFO", "NOT ENOUGH EVIDENCE": "NOT ENOUGH INFO",
    "CONFLICTING_EVIDENCE/CHERRYPICKING": "CONFLICTING",
    "CONFLICTING EVIDENCE/CHERRYPICKING": "CONFLICTING",
}


class AVeriTeCAdapter(DatasetAdapter):
    name = "AVeriTeC"

    def normalize(self, row: dict[str, Any], split: str) -> NormalizedExample:
        raw_label = first_present(row, "label", "gold_label")
        label = AVERITEC_LABELS.get(str(raw_label).upper(), raw_label) if raw_label else None
        evidence = []
        items = first_present(row, "questions", "evidence", default=[])
        for index, item in enumerate(items):
            if isinstance(item, str):
                evidence.append(EvidenceItem(str(index), item, source="web"))
                continue
            question = item.get("question", "")
            answers = item.get("answers", [])
            if isinstance(answers, dict):
                answers = [answers]
            for answer_index, answer in enumerate(answers):
                text = answer if isinstance(answer, str) else first_present(answer, "answer", "text", default="")
                if text:
                    evidence.append(EvidenceItem(
                        f"{index}:{answer_index}", f"Q: {question}\nA: {text}",
                        url=answer.get("url") if isinstance(answer, dict) else None, source="web",
                    ))
        claim_id = first_present(row, "claim_id", "id")
        return NormalizedExample(str(claim_id), self.name, split, row["claim"], label, evidence, {"claim_date": row.get("claim_date")})
