"""Cost-aware, memory-conditioned V2 routing policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .schemas import ErrorMemoryRecord


@dataclass(frozen=True)
class RoutingDecision:
    action: str
    route: bool
    expected_gain: float
    expected_cost: float
    utility: float
    support: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def select_action(
    neighbors: Iterable[tuple[ErrorMemoryRecord, float]], *,
    uncertainty: float, cost_weight: float = 0.02, min_support: int = 2,
    min_similarity: float = 0.05, min_utility: float = 0.0,
) -> RoutingDecision:
    by_action: dict[str, list[tuple[ErrorMemoryRecord, float]]] = {}
    for record, similarity in neighbors:
        if similarity >= min_similarity and record.recovered is not None:
            by_action.setdefault(record.action, []).append((record, similarity))

    candidates = []
    for action, rows in by_action.items():
        if len(rows) < min_support:
            continue
        weight = sum(similarity for _, similarity in rows)
        gain = sum(
            (record.utility if record.utility is not None else float(record.recovered)) * similarity
            for record, similarity in rows
        ) / weight
        cost = sum((record.extra_llm_calls + record.extra_tokens / 1000) * similarity for record, similarity in rows) / weight
        utility = gain * max(0.0, min(1.0, uncertainty)) - cost_weight * cost
        candidates.append((utility, action, gain, cost, len(rows)))

    if not candidates:
        return RoutingDecision("none", False, 0.0, 0.0, 0.0, 0, "insufficient compatible memory")
    utility, action, gain, cost, support = max(candidates)
    route = utility > min_utility
    return RoutingDecision(action if route else "none", route, gain, cost, utility, support, "positive memory-conditioned utility" if route else "utility below threshold")
