"""Residual-error feature extraction and descriptive analysis for B1."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .error_memory import score_distribution
from .evaluation_v2 import accuracy, macro_f1


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def token_count(text):
    return len(str(text).split())


def score_features(scores):
    probabilities = score_distribution(scores or {})
    ranked = sorted(probabilities.values(), reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else (ranked[0] if ranked else None)
    entropy = -sum(value * math.log(value) for value in probabilities.values() if value > 0)
    return probabilities, margin, entropy if probabilities else None


def transition_hypothesis(gold, predicted):
    mapping = {
        ("SUPPORTED", "REFUTED"): "false_contradiction",
        ("REFUTED", "SUPPORTED"): "missed_contradiction",
        ("CONFLICTING", "SUPPORTED"): "conflicting_mistaken_as_supported",
        ("CONFLICTING", "REFUTED"): "conflicting_mistaken_as_refuted",
        ("CONFLICTING", "NOT ENOUGH INFO"): "missed_conflicting_evidence",
        ("NOT ENOUGH INFO", "REFUTED"): "insufficient_evidence_mistaken_as_refutation",
    }
    return mapping.get((gold, predicted), "other_or_undetermined")


def build_residual_records(gold_rows, prediction_rows):
    gold, predictions = indexed(gold_rows, "gold"), indexed(prediction_rows, "predictions")
    if set(gold) != set(predictions):
        raise ValueError("Gold and prediction IDs must match exactly")
    records = []
    for key, row in gold.items():
        prediction = predictions[key]
        scores = prediction.get("metadata", {}).get("label_scores", {})
        probabilities, margin, entropy = score_features(scores)
        evidence = row.get("evidence", [])
        texts = [item.get("text", "") if isinstance(item, dict) else str(item) for item in evidence]
        expected, predicted = row["label"], prediction["prediction"]
        correct = expected == predicted
        records.append({
            "example_id": key, "dataset": row.get("dataset"), "split": row.get("split"),
            "claim": row["claim"], "gold_label": expected, "prediction": predicted,
            "correct": correct, "error_transition": None if correct else f"{expected}->{predicted}",
            "diagnosis": {
                "error_type": None if correct else transition_hypothesis(expected, predicted),
                "status": "not_applicable" if correct else "hypothesis_unvalidated",
                "basis": "observable_label_transition_only",
            },
            "label_scores": scores, "label_probabilities": probabilities,
            "prediction_margin": margin, "prediction_entropy": entropy,
            "evidence_ids": [item.get("evidence_id") for item in evidence if isinstance(item, dict)],
            "evidence_provenance": [
                {field: item.get(field) for field in ("evidence_id", "title", "url", "source")}
                for item in evidence if isinstance(item, dict)
            ],
            "claim_tokens": token_count(row["claim"]), "evidence_count": len(evidence),
            "evidence_tokens": sum(token_count(text) for text in texts),
            "latency_ms": float(prediction.get("latency_ms", 0)),
            "generated_tokens": int(prediction.get("token_count", 0)),
            "model_forward_passes": float(prediction.get("metadata", {}).get("model_forward_passes", 1)),
            "model": prediction.get("model"),
            "decoding_mode": prediction.get("metadata", {}).get("decoding_mode"),
            "instruction_profile": prediction.get("metadata", {}).get("instruction_profile"),
        })
    return records


def distribution(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}
    def quantile(probability):
        position = (len(values) - 1) * probability
        lower, upper = int(position), min(int(position) + 1, len(values) - 1)
        fraction = position - lower
        return values[lower] * (1 - fraction) + values[upper] * fraction
    return {"n": len(values), "mean": sum(values) / len(values), "median": quantile(0.5),
            "p25": quantile(0.25), "p75": quantile(0.75)}


def summarize_residuals(records):
    expected = [row["gold_label"] for row in records]
    predicted = [row["prediction"] for row in records]
    labels = sorted(set(expected) | set(predicted))
    per_class, confusion = {}, {}
    for label in labels:
        tp = sum(g == p == label for g, p in zip(expected, predicted))
        fp = sum(g != label and p == label for g, p in zip(expected, predicted))
        fn = sum(g == label and p != label for g, p in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                            "support": sum(g == label for g in expected),
                            "error_rate": fn / sum(g == label for g in expected) if any(g == label for g in expected) else 0.0}
        for output in labels:
            confusion[f"{label}->{output}"] = sum(g == label and p == output for g, p in zip(expected, predicted))
    groups = defaultdict(list)
    for row in records:
        groups["correct" if row["correct"] else "error"].append(row)
    residuals = [row for row in records if not row["correct"]]
    return {
        "n": len(records), "n_correct": len(records) - len(residuals), "n_errors": len(residuals),
        "accuracy": accuracy(predicted, expected), "macro_f1": macro_f1(predicted, expected),
        "per_class": per_class, "confusion": confusion,
        "error_transitions": dict(Counter(row["error_transition"] for row in residuals)),
        "diagnosis_hypotheses": dict(Counter(row["diagnosis"]["error_type"] for row in residuals)),
        "margin": {name: distribution(row["prediction_margin"] for row in rows) for name, rows in groups.items()},
        "entropy": {name: distribution(row["prediction_entropy"] for row in rows) for name, rows in groups.items()},
        "evidence_tokens": {name: distribution(row["evidence_tokens"] for row in rows) for name, rows in groups.items()},
        "evidence_count": {name: distribution(row["evidence_count"] for row in rows) for name, rows in groups.items()},
        "retrieval_score_association": {"status": "not_available", "reason": "normalized oracle evidence has no common retrieval score"},
        "causal_diagnosis_status": "not_validated",
    }
