"""Training-free policy replay over frozen baseline predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .error_memory import ErrorMemoryStore
from .routing_v2 import RoutingDecision, select_action


@dataclass(frozen=True)
class InterventionResult:
    prediction: str
    tokens: int = 0
    llm_calls: int = 0
    latency_ms: float = 0.0
    metadata: dict | None = None


Intervention = Callable[[dict, RoutingDecision], InterventionResult]


def replay_policy(
    rows: Iterable[dict], memory: ErrorMemoryStore, interventions: dict[str, Intervention], *,
    allowed_memory_splits: set[str], top_k: int = 5, cost_weight: float = 0.02,
    min_support: int = 2, min_similarity: float = 0.05, min_utility: float = 0.0,
) -> list[dict]:
    """Apply a policy without training or mutating memory during evaluation."""
    outputs = []
    for row in rows:
        confidence = row.get("confidence")
        uncertainty = 1.0 - float(confidence) if confidence is not None else 1.0
        neighbors = memory.retrieve(
            row["claim"], dataset=row.get("dataset"), top_k=top_k,
            allowed_splits=allowed_memory_splits,
            baseline_prediction=row["prediction"],
            label_scores=row.get("metadata", {}).get("label_scores"),
        )
        decision = select_action(
            neighbors, uncertainty=uncertainty, cost_weight=cost_weight,
            min_support=min_support, min_similarity=min_similarity, min_utility=min_utility,
        )
        output = dict(row)
        output.update({
            "baseline_prediction": row["prediction"],
            "final_prediction": row["prediction"],
            "routed": False,
            "route": decision.to_dict(),
            "llm_calls": int(row.get("llm_calls", 1)),
            "extra_tokens": 0,
        })
        if decision.route and decision.action in interventions:
            result = interventions[decision.action](row, decision)
            output.update({
                "final_prediction": result.prediction,
                "routed": True,
                "llm_calls": output["llm_calls"] + result.llm_calls,
                "extra_tokens": result.tokens,
                "intervention_latency_ms": result.latency_ms,
                "intervention_metadata": result.metadata or {},
            })
        outputs.append(output)
    return outputs
