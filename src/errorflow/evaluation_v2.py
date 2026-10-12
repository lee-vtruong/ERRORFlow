"""Unified effectiveness, safety, and efficiency metrics."""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def accuracy(predictions: Sequence[str], gold: Sequence[str]) -> float:
    return sum(p == g for p, g in zip(predictions, gold)) / len(gold) if gold else 0.0


def macro_f1(predictions: Sequence[str], gold: Sequence[str]) -> float:
    labels = sorted(set(predictions) | set(gold))
    scores = []
    for label in labels:
        tp = sum(p == g == label for p, g in zip(predictions, gold))
        fp = sum(p == label and g != label for p, g in zip(predictions, gold))
        fn = sum(p != label and g == label for p, g in zip(predictions, gold))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_run(rows: Sequence[dict]) -> dict:
    gold = [row["gold_label"] for row in rows]
    baseline = [row["baseline_prediction"] for row in rows]
    final = [row.get("final_prediction", row["baseline_prediction"]) for row in rows]
    helpful = sum(b != g and f == g for b, f, g in zip(baseline, final, gold))
    harmful = sum(b == g and f != g for b, f, g in zip(baseline, final, gold))
    routed = sum(bool(row.get("routed")) for row in rows)
    return {
        "n": len(rows),
        "baseline_accuracy": accuracy(baseline, gold),
        "final_accuracy": accuracy(final, gold),
        "baseline_macro_f1": macro_f1(baseline, gold),
        "final_macro_f1": macro_f1(final, gold),
        "helpful": helpful,
        "harmful": harmful,
        "net_recovery": helpful - harmful,
        "route_rate": routed / len(rows) if rows else 0.0,
        "avg_llm_calls": sum(float(row.get("llm_calls", 1)) for row in rows) / len(rows) if rows else 0.0,
        "label_distribution": dict(Counter(gold)),
    }
