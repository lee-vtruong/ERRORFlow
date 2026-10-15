"""Small dependency-free L2 logistic gate for disagreement arbitration."""

from __future__ import annotations

import math
from dataclasses import dataclass


FEATURE_NAMES = (
    "U_H",
    "U_N",
    "U_N_minus_U_H",
    "cross_label_similarity_mass",
    "b1_entropy",
    "neighborhood_overlap",
)


def _number(value, fallback=0.0):
    if value is None:
        return fallback
    value = float(value)
    return value if math.isfinite(value) else fallback


def feature_vector(row):
    """Extract the six preregistered pre-action features only."""
    hard = row["hard"]
    nlf = row["nlf"]
    hard_u = _number(hard.get("utility"))
    nlf_u = _number(nlf.get("utility"))
    return (
        hard_u,
        nlf_u,
        nlf_u - hard_u,
        _number(nlf.get("cross_label_similarity_mass")),
        _number(row.get("baseline_entropy")),
        _number(row.get("top5_neighbor_overlap")),
    )


def sigmoid(value):
    if value >= 0:
        z = math.exp(-min(value, 40.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -40.0))
    return z / (1.0 + z)


def _solve(matrix, vector):
    """Deterministic Gaussian elimination with a tiny diagonal safeguard."""
    n = len(vector)
    augmented = [list(matrix[i]) + [float(vector[i])] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        if abs(augmented[column][column]) < 1e-10:
            augmented[column][column] += 1e-8
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    left - factor * right
                    for left, right in zip(augmented[row], augmented[column])
                ]
    return [augmented[row][-1] for row in range(n)]


@dataclass(frozen=True)
class LogisticGate:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    l2: float
    threshold: float
    training_examples: int
    positive_examples: int

    def probability(self, values):
        standardized = [
            (float(value) - mean) / scale
            for value, mean, scale in zip(values, self.means, self.scales)
        ]
        linear = self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardized)
        )
        return sigmoid(linear)

    def to_dict(self):
        return {
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "l2": self.l2,
            "threshold": self.threshold,
            "training_examples": self.training_examples,
            "positive_examples": self.positive_examples,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            tuple(payload["feature_names"]), tuple(payload["means"]),
            tuple(payload["scales"]), tuple(payload["coefficients"]),
            float(payload["intercept"]), float(payload["l2"]),
            float(payload["threshold"]), int(payload["training_examples"]),
            int(payload["positive_examples"]),
        )


def fit_logistic(rows, labels, feature_indices, l2, threshold=0.5, max_iter=100):
    if not rows or len(rows) != len(labels):
        raise ValueError("Logistic fitting requires nonempty aligned rows and labels")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Logistic labels must be binary")
    names = tuple(FEATURE_NAMES[index] for index in feature_indices)
    selected = [[float(row[index]) for index in feature_indices] for row in rows]
    means = [sum(column) / len(selected) for column in zip(*selected)]
    scales = []
    for index, mean in enumerate(means):
        variance = sum((row[index] - mean) ** 2 for row in selected) / len(selected)
        scales.append(math.sqrt(variance) if variance > 1e-12 else 1.0)
    x = [
        [1.0] + [(value - mean) / scale for value, mean, scale in zip(row, means, scales)]
        for row in selected
    ]
    dimensions = len(x[0])
    weights = [0.0] * dimensions
    prevalence = min(max(sum(labels) / len(labels), 1e-6), 1 - 1e-6)
    weights[0] = math.log(prevalence / (1.0 - prevalence))
    for _ in range(max_iter):
        probabilities = [sigmoid(sum(w * value for w, value in zip(weights, row))) for row in x]
        gradient = [0.0] * dimensions
        hessian = [[0.0] * dimensions for _ in range(dimensions)]
        for row, label, probability in zip(x, labels, probabilities):
            error = probability - label
            curvature = max(probability * (1.0 - probability), 1e-8)
            for left in range(dimensions):
                gradient[left] += error * row[left]
                for right in range(dimensions):
                    hessian[left][right] += curvature * row[left] * row[right]
        for index in range(1, dimensions):
            gradient[index] += l2 * weights[index]
            hessian[index][index] += l2
        hessian[0][0] += 1e-8
        step = _solve(hessian, gradient)
        weights = [weight - delta for weight, delta in zip(weights, step)]
        if max(abs(delta) for delta in step) < 1e-8:
            break
    return LogisticGate(
        names, tuple(means), tuple(scales), tuple(weights[1:]), weights[0],
        float(l2), float(threshold), len(labels), sum(labels),
    )


def subset_vector(values, feature_names):
    indexes = [FEATURE_NAMES.index(name) for name in feature_names]
    return tuple(values[index] for index in indexes)
