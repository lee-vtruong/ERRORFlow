"""Similarity and utility primitives for ERRORFlow V3 substitution ablations.

The functions in this module are deliberately side-effect free.  The paper
ablation driver owns fold isolation, artifact loading, and model caching.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .error_memory import score_distribution


# Keep substitution studies bit-for-bit aligned with the established R6
# memory retriever.  The earlier exploratory Group-C runner used ``\w+`` and
# therefore split apostrophes/hyphens differently from production retrieval.
TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO", "CONFLICTING")


def tokens(text):
    return TOKEN_RE.findall(str(text).lower())


def jaccard(left, right):
    a, b = set(tokens(left)), set(tokens(right))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


def build_idf(documents):
    documents = [set(tokens(text)) for text in documents]
    n = len(documents)
    frequencies = Counter(token for document in documents for token in document)
    return {
        token: math.log((n + 1.0) / (frequency + 1.0)) + 1.0
        for token, frequency in frequencies.items()
    }


def tfidf_vector(text, idf):
    counts = Counter(tokens(text))
    if not counts:
        return {}
    total = sum(counts.values())
    return {token: count / total * idf.get(token, 0.0) for token, count in counts.items()}


def sparse_cosine(left, right):
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def bm25_prepare(documents, k1=1.2, b=0.75):
    tokenized = [tokens(text) for text in documents]
    n = len(tokenized)
    average_length = sum(map(len, tokenized)) / n if n else 0.0
    frequencies = Counter(token for document in tokenized for token in set(document))
    idf = {
        token: math.log(1.0 + (n - frequency + 0.5) / (frequency + 0.5))
        for token, frequency in frequencies.items()
    }
    return [Counter(document) for document in tokenized], idf, average_length, k1, b


def bm25_score(query_text, document_counts, idf, average_length, k1=1.2, b=0.75):
    length = sum(document_counts.values())
    norm = 1.0 - b + b * length / average_length if average_length else 1.0
    score = 0.0
    for token in set(tokens(query_text)):
        frequency = document_counts.get(token, 0)
        if frequency:
            score += idf.get(token, 0.0) * frequency * (k1 + 1.0) / (frequency + k1 * norm)
    return score


def saturate(value):
    """Map a non-negative unbounded retrieval score to [0, 1)."""
    value = max(0.0, float(value))
    return value / (1.0 + value)


def probabilities(scores):
    distribution = score_distribution(scores or {})
    return [float(distribution.get(label, 0.0)) for label in LABELS]


def label_similarity(left_scores, right_scores, metric="l1"):
    left, right = probabilities(left_scores), probabilities(right_scores)
    if metric == "l1":
        return max(0.0, 1.0 - sum(abs(a - b) for a, b in zip(left, right)) / 2.0)
    if metric == "cosine":
        dot = sum(a * b for a, b in zip(left, right))
        norm = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
        return dot / norm if norm else 0.0
    if metric == "l2":
        return max(0.0, 1.0 - math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right))) / math.sqrt(2.0))
    if metric == "jensen_shannon":
        middle = [(a + b) / 2.0 for a, b in zip(left, right)]
        def kl(values, reference):
            return sum(value * math.log(value / base) for value, base in zip(values, reference) if value > 0 and base > 0)
        divergence = 0.5 * kl(left, middle) + 0.5 * kl(right, middle)
        return max(0.0, 1.0 - divergence / math.log(2.0))
    if metric == "hellinger":
        distance = math.sqrt(sum((math.sqrt(a) - math.sqrt(b)) ** 2 for a, b in zip(left, right))) / math.sqrt(2.0)
        return max(0.0, 1.0 - distance)
    raise ValueError(f"Unknown label similarity metric: {metric}")


def evidence_length_similarity(left, right):
    return 1.0 / (1.0 + abs(float(left or 0.0) - float(right or 0.0)) / 100.0)


def scalar_similarity(left, right):
    return max(0.0, 1.0 - abs(float(left) - float(right)))


def aggregate_similarity(components, weights=(0.35, 0.50, 0.15), mode="weighted_sum", epsilon=1e-9):
    if len(components) != len(weights):
        raise ValueError("components and weights must have equal length")
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    normalized = [weight / total for weight in weights]
    if mode == "weighted_sum":
        return sum(weight * value for weight, value in zip(normalized, components))
    if mode == "multiplicative":
        return math.exp(sum(weight * math.log(max(epsilon, value)) for weight, value in zip(normalized, components)))
    raise ValueError(f"Unknown aggregation mode: {mode}")


def estimate_utility(neighbors, mode="smoothed", cost_weight=0.1, prior=None):
    """Estimate recovery minus regression minus cost from weighted outcomes.

    ``neighbors`` contains ``(record, similarity)`` pairs.  ``prior`` is a
    mapping with recovery/regression rates and strength for Bayesian shrinkage.
    """
    if not neighbors:
        return None
    weights = [max(0.0, float(similarity)) for _, similarity in neighbors]
    weight = sum(weights)
    if weight <= 1e-12:
        weights = [1.0] * len(neighbors)
        weight = float(len(neighbors))
    recovery = sum(record.recovery_count * value for (record, _), value in zip(neighbors, weights))
    regression = sum(record.regression_count * value for (record, _), value in zip(neighbors, weights))
    cost = sum(float(record.estimated_extra_cost or 0.0) * value for (record, _), value in zip(neighbors, weights)) / weight
    if mode == "smoothed":
        recovery_rate = (recovery + 1.0) / (weight + 2.0)
        regression_rate = (regression + 1.0) / (weight + 2.0)
    elif mode == "unsmoothed":
        recovery_rate, regression_rate = recovery / weight, regression / weight
    elif mode == "bayesian":
        prior = prior or {"recovery": 0.0, "regression": 0.0, "strength": 2.0}
        strength = float(prior.get("strength", 2.0))
        recovery_rate = (recovery + strength * float(prior["recovery"])) / (weight + strength)
        regression_rate = (regression + strength * float(prior["regression"])) / (weight + strength)
    elif mode == "effective_sample_size":
        ess = weight * weight / sum(value * value for value in weights)
        pseudo = 1.0 / math.sqrt(max(ess, 1.0))
        recovery_rate = (recovery + pseudo) / (weight + 2.0 * pseudo)
        regression_rate = (regression + pseudo) / (weight + 2.0 * pseudo)
    elif mode == "recovery_only":
        recovery_rate = (recovery + 1.0) / (weight + 2.0)
        regression_rate = 0.0
    elif mode == "regression_only":
        # A high score denotes a region with low historical regression risk.
        recovery_rate = 1.0
        regression_rate = (regression + 1.0) / (weight + 2.0)
    else:
        raise ValueError(f"Unknown utility estimator: {mode}")
    return {
        "utility": recovery_rate - regression_rate - cost_weight * cost,
        "recovery_rate": recovery_rate,
        "regression_rate": regression_rate,
        "cost": cost,
        "support": len(neighbors),
        "weight": weight,
    }


def confidence_bin(probability, width=0.2):
    return min(int(max(0.0, min(1.0, float(probability))) / width), int(1.0 / width) - 1)


def margin_bin(margin, width=0.1):
    return min(int(max(0.0, min(1.0, float(margin or 0.0))) / width), int(1.0 / width) - 1)
