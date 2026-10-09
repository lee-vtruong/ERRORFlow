"""Serializable contracts for verifier predictions and execution traces."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PredictionRecord:
    claim_id: str
    claim: str
    prediction: str
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float | None = None
    model: str = "unknown"
    checkpoint: str = "unknown"
    seed: int | None = None
    split: str = "unknown"
    token_count: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionTrace:
    claim_id: str
    steps: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    total_tokens: int = 0
    llm_calls: int = 0
    latency_ms: float = 0.0
    official_validation_used: bool = False
    test_split_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
