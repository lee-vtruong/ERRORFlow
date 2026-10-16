"""Fold-safe supervised routing controls for ERRORFlow V3.

The controls predict the observed signed effect of replacing Base with Alt.
Grouped source evaluation is cross-fitted by the frozen outer folds. External
evaluation fits once on the complete source population and never reads target
labels until predictions and routes have been materialized.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
from pathlib import Path

import numpy as np

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap, percentile


LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO", "CONFLICTING")


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def gold_label(row):
    value = row.get("label", row.get("gold_label"))
    if value is None:
        raise ValueError(f"Missing gold label for {row.get('example_id')}")
    return str(value)


def scores(row):
    value = row.get("label_scores") or row.get("metadata", {}).get("label_scores")
    if not value:
        raise ValueError(f"Missing label scores for {row.get('example_id')}")
    return {str(key): float(number) for key, number in value.items()}


def distribution(row):
    raw = scores(row)
    peak = max(raw.values())
    exp = {key: math.exp(value - peak) for key, value in raw.items()}
    total = sum(exp.values())
    return {key: value / total for key, value in exp.items()}


def evidence_text(example):
    return " ".join(str(item.get("text", "")) for item in example.get("evidence", []))


def feature_vector(example, baseline, interactions=False):
    probs = distribution(baseline)
    values = [probs.get(label, 0.0) for label in LABELS]
    predicted = str(baseline["prediction"])
    one_hot = [float(predicted == label) for label in LABELS]
    ordered = sorted(values, reverse=True)
    confidence = ordered[0]
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    entropy = -sum(value * math.log(max(value, 1e-300)) for value in values)
    claim_tokens = len(str(example.get("claim", "")).split())
    evidence_count = len(example.get("evidence", []))
    evidence_tokens = len(evidence_text(example).split())
    result = values + one_hot + [
        confidence, margin, entropy, math.log1p(claim_tokens),
        math.log1p(evidence_count), math.log1p(evidence_tokens),
    ]
    if interactions:
        result.extend(confidence * value for value in one_hot)
        result.extend(margin * value for value in one_hot)
        result.extend([
            confidence * math.log1p(evidence_tokens),
            margin * math.log1p(claim_tokens),
        ])
    return result


def signed_outcome(gold, baseline, candidate):
    before = baseline == gold
    after = candidate == gold
    return 1 if (not before and after) else (-1 if (before and not after) else 0)


def fit_softmax(x, outcomes, l2=0.01, epochs=800, learning_rate=0.05,
                class_weight=None):
    x = np.asarray(x, dtype=np.float64)
    means, scales = x.mean(0), x.std(0)
    scales[scales < 1e-8] = 1.0
    design = np.column_stack([np.ones(len(x)), (x - means) / scales])
    mapping = {-1: 0, 0: 1, 1: 2}
    target_ids = np.asarray([mapping[value] for value in outcomes], dtype=int)
    target = np.eye(3)[target_ids]
    sample_weight = np.ones(len(target_ids), dtype=np.float64)
    if class_weight == "balanced":
        counts = np.bincount(target_ids, minlength=3)
        for class_id, count in enumerate(counts):
            if count:
                sample_weight[target_ids == class_id] = len(target_ids) / (3.0 * count)
    elif class_weight is not None:
        raise ValueError(f"Unsupported class_weight: {class_weight}")
    weights = np.zeros((design.shape[1], 3), dtype=np.float64)
    for _ in range(epochs):
        logits = design @ weights
        logits -= logits.max(axis=1, keepdims=True)
        predicted = np.exp(logits)
        predicted /= predicted.sum(axis=1, keepdims=True)
        gradient = design.T @ ((predicted - target) * sample_weight[:, None]) / sample_weight.sum()
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return weights, means, scales


def predict_softmax(model, x):
    weights, means, scales = model
    x = np.asarray(x, dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), (x - means) / scales])
    logits = design @ weights
    logits -= logits.max(axis=1, keepdims=True)
    predicted = np.exp(logits)
    predicted /= predicted.sum(axis=1, keepdims=True)
    return predicted[:, 2] - predicted[:, 0]


def fit_boosted_stumps(x, outcomes, rounds=50, learning_rate=0.1, bins=12):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    initial = float(y.mean())
    predicted = np.full(len(y), initial, dtype=np.float64)
    stumps = []
    for _ in range(rounds):
        residual = y - predicted
        best = None
        for feature in range(x.shape[1]):
            thresholds = np.unique(np.quantile(x[:, feature], np.linspace(0.05, 0.95, bins)))
            for threshold in thresholds:
                left = x[:, feature] <= threshold
                if not left.any() or left.all():
                    continue
                left_value = float(residual[left].mean())
                right_value = float(residual[~left].mean())
                error = float(((residual[left] - left_value) ** 2).sum() +
                              ((residual[~left] - right_value) ** 2).sum())
                candidate = (error, feature, float(threshold), left_value, right_value)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        _, feature, threshold, left_value, right_value = best
        update = np.where(x[:, feature] <= threshold, left_value, right_value)
        predicted += learning_rate * update
        stumps.append((feature, threshold, left_value, right_value))
    return initial, learning_rate, stumps


def predict_boosted_stumps(model, x):
    initial, learning_rate, stumps = model
    x = np.asarray(x, dtype=np.float64)
    predicted = np.full(len(x), initial, dtype=np.float64)
    for feature, threshold, left_value, right_value in stumps:
        predicted += learning_rate * np.where(
            x[:, feature] <= threshold, left_value, right_value
        )
    return predicted


def top_count(ids, values, count):
    ranked = sorted(ids, key=lambda key: (values[key], key), reverse=True)
    return set(ranked[:count])


def per_label_rank_scores(train_ids, train_values, test_ids, test_values, baseline):
    """Calibrate a score to its source empirical percentile within Base label.

    This is a label-conditional threshold control.  It uses no held-out labels:
    only source-fold model scores and Base predictions define each empirical CDF.
    """
    pools = {}
    global_pool = sorted(float(train_values[key]) for key in train_ids)
    for key in train_ids:
        label = str(baseline[key]["prediction"])
        pools.setdefault(label, []).append(float(train_values[key]))
    for values in pools.values():
        values.sort()
    result = {}
    for key in test_ids:
        pool = pools.get(str(baseline[key]["prediction"]), global_pool)
        result[key] = bisect.bisect_right(pool, float(test_values[key])) / len(pool)
    return result


def project(label, enabled):
    return "NOT ENOUGH INFO" if enabled and label == "CONFLICTING" else label


def evaluate(ids, gold, baseline, candidate, selected, projection):
    expected = [gold_label(gold[key]) for key in ids]
    before = [project(baseline[key]["prediction"], projection) for key in ids]
    after = [project(candidate[key]["prediction"], projection) for key in ids]
    prediction = [right if key in selected else left for key, left, right in zip(ids, before, after)]
    helpful = sum(left != target and pred == target for target, left, pred in zip(expected, before, prediction))
    harmful = sum(left == target and pred != target for target, left, pred in zip(expected, before, prediction))
    per_class = {}
    for label in sorted(set(expected) | set(prediction)):
        tp = sum(g == p == label for g, p in zip(expected, prediction))
        fp = sum(g != label and p == label for g, p in zip(expected, prediction))
        fn = sum(g == label and p != label for g, p in zip(expected, prediction))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": sum(g == label for g in expected),
        }
    return {
        "routed": len(selected), "route_rate": len(selected) / len(ids),
        "accuracy": accuracy(prediction, expected), "macro_f1": macro_f1(prediction, expected),
        "helpful": helpful, "harmful": harmful, "net": helpful - harmful,
        "per_class": per_class,
        "avg_model_forward_passes": 4.0 + 4.0 * len(selected) / len(ids),
    }, prediction


def holm_adjust(values):
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    adjusted, running = {}, 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (len(ordered) - index)))
        adjusted[name] = running
    return adjusted


def paired_bootstrap_interval(gold, left, right, samples, seed, alpha):
    """Percentile interval with caller-specified two-sided alpha."""
    rng = random.Random(seed)
    accuracy_deltas, f1_deltas = [], []
    for _ in range(samples):
        indexes = [rng.randrange(len(gold)) for _ in gold]
        sampled_gold = [gold[index] for index in indexes]
        sampled_left = [left[index] for index in indexes]
        sampled_right = [right[index] for index in indexes]
        accuracy_deltas.append(
            accuracy(sampled_right, sampled_gold) - accuracy(sampled_left, sampled_gold)
        )
        f1_deltas.append(
            macro_f1(sampled_right, sampled_gold) - macro_f1(sampled_left, sampled_gold)
        )
    return {
        "two_sided_alpha": alpha,
        "confidence_level": 1.0 - alpha,
        "accuracy_delta_ci": [
            percentile(accuracy_deltas, alpha / 2),
            percentile(accuracy_deltas, 1.0 - alpha / 2),
        ],
        "macro_f1_delta_ci": [
            percentile(f1_deltas, alpha / 2),
            percentile(f1_deltas, 1.0 - alpha / 2),
        ],
    }


def load_nlf(meta_dir, ids):
    rows = [row for row in read_jsonl(meta_dir / "per_query_predictions.jsonl")
            if row.get("mode") == "natural" and row.get("policy") == "D1_NLF"]
    index = indexed(rows, "natural Memory-Global predictions")
    if set(index) != set(ids):
        raise ValueError("Memory-Global IDs do not match target IDs")
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--meta-dir", required=True, type=Path)
    parser.add_argument("--protocol", choices=("grouped_internal", "external"), required=True)
    parser.add_argument("--source-gold", type=Path)
    parser.add_argument("--source-baseline", type=Path)
    parser.add_argument("--source-candidate", type=Path)
    parser.add_argument("--source-meta-dir", type=Path)
    parser.add_argument("--route-count", type=int)
    parser.add_argument("--project-conflicting-to-nei", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    gold = indexed(read_jsonl(args.gold), "gold")
    baseline = indexed(read_jsonl(args.baseline), "baseline")
    candidate = indexed(read_jsonl(args.candidate), "candidate")
    ids = sorted(gold)
    if set(ids) != set(baseline) or set(ids) != set(candidate):
        raise ValueError("Target gold/Base/Alt IDs must match")
    nlf = load_nlf(args.meta_dir, ids)
    route_count = args.route_count if args.route_count is not None else sum(bool(nlf[key]["routed"]) for key in ids)

    target_features = {
        False: {key: feature_vector(gold[key], baseline[key], False) for key in ids},
        True: {key: feature_vector(gold[key], baseline[key], True) for key in ids},
    }
    model_scores = {name: {} for name in (
        "Logistic", "Logistic-BalancedOutcome", "Logistic-PerLabelRank",
        "Logistic-Interactions", "Boosted-Stumps",
    )}

    if args.protocol == "grouped_internal":
        fold_rows = indexed(read_jsonl(args.meta_dir / "per_query_features.jsonl"), "fold features")
        folds = {key: int(fold_rows[key]["outer_fold"]) for key in ids}
        training_source = (gold, baseline, candidate)
        for held_out in sorted(set(folds.values())):
            train_ids = [key for key in ids if folds[key] != held_out]
            test_ids = [key for key in ids if folds[key] == held_out]
            outcomes = [signed_outcome(gold_label(gold[key]), baseline[key]["prediction"], candidate[key]["prediction"]) for key in train_ids]
            for interactions, name in ((False, "Logistic"), (True, "Logistic-Interactions")):
                model = fit_softmax([target_features[interactions][key] for key in train_ids], outcomes)
                values = predict_softmax(model, [target_features[interactions][key] for key in test_ids])
                model_scores[name].update(zip(test_ids, map(float, values)))
                if name == "Logistic":
                    train_values = predict_softmax(
                        model, [target_features[False][key] for key in train_ids]
                    )
                    train_scores = dict(zip(train_ids, map(float, train_values)))
                    test_scores = dict(zip(test_ids, map(float, values)))
                    model_scores["Logistic-PerLabelRank"].update(per_label_rank_scores(
                        train_ids, train_scores, test_ids, test_scores, baseline
                    ))
            model = fit_softmax(
                [target_features[False][key] for key in train_ids], outcomes,
                class_weight="balanced",
            )
            values = predict_softmax(model, [target_features[False][key] for key in test_ids])
            model_scores["Logistic-BalancedOutcome"].update(zip(test_ids, map(float, values)))
            model = fit_boosted_stumps([target_features[False][key] for key in train_ids], outcomes)
            values = predict_boosted_stumps(model, [target_features[False][key] for key in test_ids])
            model_scores["Boosted-Stumps"].update(zip(test_ids, map(float, values)))
        target_labels_used_for_fitting = False
    else:
        required = (args.source_gold, args.source_baseline, args.source_candidate, args.source_meta_dir)
        if not all(required):
            parser.error("external requires all --source-* arguments")
        source_gold = indexed(read_jsonl(args.source_gold), "source gold")
        source_baseline = indexed(read_jsonl(args.source_baseline), "source baseline")
        source_candidate = indexed(read_jsonl(args.source_candidate), "source candidate")
        source_ids = sorted(source_gold)
        if set(source_ids) != set(source_baseline) or set(source_ids) != set(source_candidate):
            raise ValueError("Source gold/Base/Alt IDs must match")
        source_features = {
            flag: {key: feature_vector(source_gold[key], source_baseline[key], flag) for key in source_ids}
            for flag in (False, True)
        }
        outcomes = [signed_outcome(gold_label(source_gold[key]), source_baseline[key]["prediction"], source_candidate[key]["prediction"]) for key in source_ids]
        for interactions, name in ((False, "Logistic"), (True, "Logistic-Interactions")):
            model = fit_softmax([source_features[interactions][key] for key in source_ids], outcomes)
            values = predict_softmax(model, [target_features[interactions][key] for key in ids])
            model_scores[name].update(zip(ids, map(float, values)))
            if name == "Logistic":
                source_values = predict_softmax(
                    model, [source_features[False][key] for key in source_ids]
                )
                source_scores = dict(zip(source_ids, map(float, source_values)))
                target_scores = dict(zip(ids, map(float, values)))
                model_scores["Logistic-PerLabelRank"].update(per_label_rank_scores(
                    source_ids, source_scores, ids, target_scores,
                    {**source_baseline, **baseline},
                ))
        model = fit_softmax(
            [source_features[False][key] for key in source_ids], outcomes,
            class_weight="balanced",
        )
        values = predict_softmax(model, [target_features[False][key] for key in ids])
        model_scores["Logistic-BalancedOutcome"].update(zip(ids, map(float, values)))
        model = fit_boosted_stumps([source_features[False][key] for key in source_ids], outcomes)
        values = predict_boosted_stumps(model, [target_features[False][key] for key in ids])
        model_scores["Boosted-Stumps"].update(zip(ids, map(float, values)))
        target_labels_used_for_fitting = False

    expected = [gold_label(gold[key]) for key in ids]
    nlf_predictions = [project(nlf[key]["prediction"], args.project_conflicting_to_nei) for key in ids]
    results, predictions, comparisons, predictions_by_name = {}, [], {}, {}
    for name, values in model_scores.items():
        selected = top_count(ids, values, route_count)
        result, predicted = evaluate(ids, gold, baseline, candidate, selected, args.project_conflicting_to_nei)
        results[name] = result
        predictions_by_name[name] = predicted
        comparisons[f"{name}_vs_Memory-Global"] = {
            "orientation": f"{name}_minus_Memory-Global",
            "mcnemar_exact": mcnemar_exact(expected, nlf_predictions, predicted),
            "paired_bootstrap": paired_bootstrap(
                expected, nlf_predictions, predicted, args.bootstrap_samples, args.seed
            ),
        }
        for key, prediction in zip(ids, predicted):
            predictions.append({
                "example_id": key, "method": name, "routed": key in selected,
                "score": values[key], "prediction": prediction,
                "gold_label": gold_label(gold[key]),
            })

    nlf_selected = {key for key in ids if bool(nlf[key]["routed"])}
    nlf_result, _ = evaluate(ids, gold, baseline, candidate, nlf_selected, args.project_conflicting_to_nei)
    raw_p = {
        name: value["mcnemar_exact"]["exact_two_sided_p"]
        for name, value in comparisons.items()
    }
    adjusted = holm_adjust(raw_p)
    for name, value in comparisons.items():
        value["holm_accuracy_p"] = adjusted[name]
    family_alpha = 0.05 / len(comparisons)
    for comparison_name, value in comparisons.items():
        method = comparison_name.removesuffix("_vs_Memory-Global")
        value["bonferroni_familywise_interval"] = paired_bootstrap_interval(
            expected, nlf_predictions, predictions_by_name[method],
            args.bootstrap_samples, args.seed, family_alpha,
        )
    report = {
        "evaluation": "fold_safe_supervised_routing_controls",
        "protocol": args.protocol, "n": len(ids), "route_count": route_count,
        "features": "Base label probabilities, predicted-label one-hot, confidence, margin, entropy, claim/evidence sizes",
        "models": {
            "Logistic": "three-class softmax outcome model; fixed L2=0.01",
            "Logistic-BalancedOutcome": "same model with inverse-frequency outcome-class weights fitted within each source fold",
            "Logistic-PerLabelRank": "plain logistic score converted to a source empirical percentile within each Base-predicted label",
            "Logistic-Interactions": "Logistic plus confidence/margin-by-label and length interactions",
            "Boosted-Stumps": "50 squared-error boosting rounds, depth-one stumps, learning rate 0.1",
        },
        "target_labels_used_for_fitting_or_selection": target_labels_used_for_fitting,
        "multiple_comparisons": {
            "family": sorted(results),
            "holm_applied_to": "exact McNemar accuracy p-values",
            "interval_adjustment": "Bonferroni percentile intervals across the complete supervised-control family",
            "familywise_alpha": 0.05,
        },
        "results": {"Memory-Global": nlf_result, **results},
        "comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", report)
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
