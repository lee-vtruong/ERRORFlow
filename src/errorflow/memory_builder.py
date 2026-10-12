"""Build structured memory from paired frozen-baseline experiments."""

from __future__ import annotations

from typing import Iterable

from .error_memory import memory_id
from .schemas import ErrorMemoryRecord

EVALUATION_SPLITS = {"dev", "development", "validation", "test"}


def build_memory_records(
    rows: Iterable[dict], *, dataset: str, source_split: str, action: str,
    permit_evaluation_routing: bool = False,
) -> list[ErrorMemoryRecord]:
    """Convert paired outcomes to memory without silently enabling leakage."""
    allow = permit_evaluation_routing or source_split.lower() not in EVALUATION_SPLITS
    records = []
    for row in rows:
        claim_id = str(row.get("claim_id", row.get("example_id")))
        baseline = row["baseline_prediction"]
        gold = row.get("gold_label")
        final = row.get("final_prediction")
        if gold is not None and baseline == gold:
            continue
        error_type = row.get("error_type") or (
            f"{str(gold).lower().replace(' ', '_')}__predicted__{str(baseline).lower().replace(' ', '_')}"
            if gold is not None else "unknown_error"
        )
        recovered = final == gold if final is not None and gold is not None else None
        records.append(ErrorMemoryRecord(
            memory_id=memory_id(dataset, claim_id, error_type, action), dataset=dataset,
            source_split=source_split, claim_id=claim_id, claim=row["claim"],
            error_type=error_type, baseline_prediction=baseline,
            diagnosis=row.get("diagnosis", error_type), action=action,
            gold_label=gold, final_prediction=final, recovered=recovered,
            utility=(1.0 if recovered else 0.0) if recovered is not None else None,
            extra_tokens=int(row.get("extra_tokens", 0)),
            extra_llm_calls=max(0, int(row.get("extra_llm_calls", row.get("llm_calls", 1) - 1))),
            latency_ms=float(row.get("intervention_latency_ms", 0.0)), allow_for_routing=allow,
            metadata={"builder": "paired_outcome_v2", "source_run_id": row.get("run_id"),
                      "evaluation_memory_override": bool(permit_evaluation_routing and source_split.lower() in EVALUATION_SPLITS)},
        ))
    return records
