"""A transparent, training-free intervention router baseline."""

from dataclasses import dataclass

from .schemas import ErrorMemoryEntry


@dataclass(frozen=True)
class RouteDecision:
    action: str
    expected_recovery: float
    estimated_cost: int
    score: float
    reason: str


def route(
    error_type: str,
    memories: list[ErrorMemoryEntry],
    *,
    cost_weight: float = 0.001,
    min_support: int = 1,
) -> RouteDecision:
    """Choose the action with best empirical recovery-minus-cost score.

    This is deliberately a simple, auditable baseline. It only uses supplied
    memory records and never inspects validation/test labels implicitly.
    """
    candidates = [
        entry for entry in memories
        if entry.error_type == error_type
        and entry.best_action
        and entry.recovery_success is not None
    ]
    stats: dict[str, list[float]] = {}
    costs: dict[str, list[int]] = {}
    for entry in candidates:
        stats.setdefault(entry.best_action or "", []).append(float(entry.recovery_success))
        costs.setdefault(entry.best_action or "", []).append(entry.extra_tokens)

    valid = {
        action: values for action, values in stats.items() if len(values) >= min_support
    }
    if not valid:
        return RouteDecision("none", 0.0, 0, 0.0, "no supported intervention in memory")

    scored = []
    for action, values in valid.items():
        recovery = sum(values) / len(values)
        cost = round(sum(costs[action]) / len(costs[action]))
        scored.append((recovery - cost_weight * cost, action, recovery, cost))
    score, action, recovery, cost = max(scored)
    return RouteDecision(
        action,
        recovery,
        cost,
        score,
        f"selected from {len(candidates)} historical {error_type} records",
    )
