"""Stable, dependency-free contracts for legacy and V2 artifacts."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ErrorMemoryEntry:
    claim_id: str
    error_type: str
    baseline_prediction: str
    gold_label: str | None = None
    diagnosis: str = ""
    candidate_actions: list[str] = field(default_factory=list)
    best_action: str | None = None
    recovery_success: bool | None = None
    extra_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceItem:
    """Dataset-neutral evidence unit."""

    evidence_id: str
    text: str
    title: str | None = None
    url: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedExample:
    """Canonical example emitted by every V2 dataset adapter."""

    example_id: str
    dataset: str
    split: str
    claim: str
    label: str | None
    evidence: list[EvidenceItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorMemoryRecord:
    """Versioned record of an observed error and intervention outcome.

    ``gold_label`` is optional so memory can also retain unsupervised diagnoses.
    ``source_split`` and ``allow_for_routing`` enforce contamination boundaries.
    """

    memory_id: str
    dataset: str
    source_split: str
    claim_id: str
    claim: str
    error_type: str
    baseline_prediction: str
    diagnosis: str
    action: str
    gold_label: str | None = None
    final_prediction: str | None = None
    recovered: bool | None = None
    regressed: bool | None = None
    utility: float | None = None
    extra_tokens: int = 0
    extra_llm_calls: int = 0
    latency_ms: float = 0.0
    embedding: list[float] | None = None
    allow_for_routing: bool = True
    schema_version: int = 2
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
