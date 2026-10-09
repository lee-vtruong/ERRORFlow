"""Stable data contracts for ERRORFlow artifacts.

The schema is intentionally dependency-free at the skeleton stage so that
dataset/model integrations can be added without locking the initial project
to a particular inference stack.
"""

from dataclasses import dataclass, field
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
