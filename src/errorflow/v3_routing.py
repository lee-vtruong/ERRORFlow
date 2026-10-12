"""Transparent matched-budget routing policies for cached V3 actions."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ActionEstimate:
    action: str
    expected_recovery: float
    expected_regression: float
    expected_cost: float
    utility: float
    support: int

    def to_dict(self):
        return asdict(self)


def estimate_action(action, neighbors, cost_weight=0.1, smoothing=1.0):
    rows = [(record, similarity) for record, similarity in neighbors if record.correction_strategy == action]
    if not rows:
        return ActionEstimate(action, 0.0, 0.0, 0.0, float("-inf"), 0)
    weight = sum(similarity for _, similarity in rows)
    denominator = weight + 2 * smoothing
    recovery = (sum(record.recovery_count * similarity for record, similarity in rows) + smoothing) / denominator
    regression = (sum(record.regression_count * similarity for record, similarity in rows) + smoothing) / denominator
    cost = sum((record.estimated_extra_cost or 0.0) * similarity for record, similarity in rows) / weight
    return ActionEstimate(action, recovery, regression, cost,
                          recovery - regression - cost_weight * cost, len(rows))


def best_memory_action(actions, neighbors, cost_weight=0.1, min_support=2, utility_aware=True):
    estimates = [estimate_action(action, neighbors, cost_weight if utility_aware else 0.0) for action in actions]
    valid = [estimate for estimate in estimates if estimate.support >= min_support]
    if not valid:
        return None, estimates
    best = max(valid, key=lambda estimate: (estimate.utility, estimate.action))
    return best, estimates


def deterministic_random_score(example_id, seed):
    digest = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16 ** 16 - 1)


def select_budget(ids, scores, budget):
    if not 0.0 <= budget <= 1.0:
        raise ValueError("budget must be between zero and one")
    count = round(len(ids) * budget)
    ranked = sorted(ids, key=lambda key: (-scores.get(key, float("-inf")), key))
    return set(ranked[:count])


def select_weighted_budget(ids, scores, weights, budget):
    if not 0.0 <= budget <= 1.0:
        raise ValueError("budget must be between zero and one")
    target = sum(weights[key] for key in ids) * budget
    ranked = sorted(ids, key=lambda key: (-scores.get(key, float("-inf")), key))
    selected, used = set(), 0.0
    for key in ranked:
        if used + weights[key] > target:
            continue
        if used >= target:
            break
        selected.add(key)
        used += weights[key]
    return selected


def select_label_conditional_budget(ids, labels, scores, budget):
    selected = set()
    for label in sorted(set(labels.values())):
        members = [key for key in ids if labels[key] == label]
        selected |= select_budget(members, scores, budget)
    return selected


def select_label_conditional_weighted_budget(ids, labels, scores, weights, budget):
    selected = set()
    for label in sorted(set(labels.values())):
        members = [key for key in ids if labels[key] == label]
        selected |= select_weighted_budget(members, scores, weights, budget)
    return selected
