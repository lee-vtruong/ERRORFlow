"""Nested P2 comparators for Group C: tuned weights and a trained router."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap
from errorflow.v3_substitution import LABELS, probabilities

try:
    from scripts.evaluate_v3_matched_budget import evaluate_selection, holm_adjust, indexed, prepare_grouped_artifacts
    from scripts.evaluate_v3_substitution_ablation import Engine, REFERENCE, fold_analysis, top_count
except ModuleNotFoundError:
    from evaluate_v3_matched_budget import evaluate_selection, holm_adjust, indexed, prepare_grouped_artifacts
    from evaluate_v3_substitution_ablation import Engine, REFERENCE, fold_analysis, top_count


def weight_grid(step=0.25):
    units = round(1.0 / step)
    return [
        (a / units, b / units, (units - a - b) / units)
        for a in range(units + 1) for b in range(units - a + 1)
        if a + b <= units
    ]


def feature_vector(row):
    distribution = probabilities(row.get("label_scores", {}))
    predicted = [float(row["prediction"] == label) for label in LABELS]
    return distribution + predicted + [
        float(row.get("prediction_margin") or 0.0),
        float(row.get("prediction_entropy") or 0.0),
        math.log1p(float(row.get("claim_tokens") or 0.0)),
        math.log1p(float(row.get("evidence_count") or 0.0)),
        math.log1p(float(row.get("evidence_tokens") or 0.0)),
    ]


def fit_softmax(features, outcomes, seed=2026, epochs=800, learning_rate=0.05, l2=0.01):
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("C6-4 requires numpy") from error
    x = np.asarray(features, dtype=np.float64)
    x = np.column_stack([np.ones(len(x)), x])
    means, scales = x[:, 1:].mean(0), x[:, 1:].std(0)
    scales[scales < 1e-8] = 1.0
    x[:, 1:] = (x[:, 1:] - means) / scales
    mapping = {-1: 0, 0: 1, 1: 2}
    y = np.asarray([mapping[value] for value in outcomes], dtype=int)
    target = np.eye(3)[y]
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.001, size=(x.shape[1], 3))
    for _ in range(epochs):
        logits = x @ weights
        logits -= logits.max(axis=1, keepdims=True)
        predicted = np.exp(logits)
        predicted /= predicted.sum(axis=1, keepdims=True)
        gradient = x.T @ (predicted - target) / len(x)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return weights, means, scales


def predict_utility(model, rows, cost_weight):
    import numpy as np
    weights, means, scales = model
    x = np.asarray([feature_vector(row) for row in rows], dtype=np.float64)
    x = np.column_stack([np.ones(len(x)), (x - means) / scales])
    logits = x @ weights
    logits -= logits.max(axis=1, keepdims=True)
    distribution = np.exp(logits)
    distribution /= distribution.sum(axis=1, keepdims=True)
    return distribution[:, 2] - distribution[:, 0] - cost_weight


def outcome(row, candidate):
    before = row["prediction"] == row["gold_label"]
    after = candidate["prediction"] == row["gold_label"]
    return 1 if not before and after else (-1 if before and not after else 0)


def evaluate_and_rows(name, ids, baseline, candidate, scores, matched_count, threshold):
    natural = {key for key in ids if scores[key] > threshold}
    matched = top_count(ids, scores, matched_count)
    results, rows, routes, cache = {}, [], [], {}
    for mode, selected in (("natural", natural), ("matched_budget", matched)):
        metrics, predictions = evaluate_selection(ids, baseline, candidate, selected)
        results[f"{name}:{mode}"] = metrics
        cache[mode] = predictions
        routes.append({"variant": name, "mode": mode, "selected_ids": sorted(selected)})
        for key, prediction in zip(ids, predictions):
            rows.append({
                "example_id": key, "variant": name, "mode": mode,
                "outer_fold": baseline[key]["outer_fold"], "routed": key in selected,
                "gold_label": baseline[key]["gold_label"], "prediction": prediction,
                "estimated_utility": float(scores[key]),
            })
    return results, rows, routes, cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--reference-summary", required=True, type=Path)
    parser.add_argument("--reference-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--action", default="conflicting_rescue_v1")
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--group-threshold", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--weight-grid-step", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    baseline_rows = list(read_jsonl(args.baseline_records))
    memory_rows = list(read_jsonl(args.memory))
    baseline_rows, records, _, grouping = prepare_grouped_artifacts(
        baseline_rows, memory_rows, args.group_threshold, args.folds, args.seed,
    )
    baseline = indexed(baseline_rows, "baseline")
    candidate = indexed(read_jsonl(args.candidate), "candidate")
    examples = indexed(read_jsonl(args.examples), "examples")
    ids = sorted(baseline)
    if set(ids) != set(candidate) or set(ids) != set(examples):
        raise ValueError("Baseline, candidate, and examples IDs must match")
    reference_summary = json.loads(args.reference_summary.read_text(encoding="utf-8"))
    matched_count = int(reference_summary["configuration"]["matched_route_count"])
    reference_rows = [
        row for row in read_jsonl(args.reference_predictions)
        if row["variant"] == "C0_full_r6_reference" and row["mode"] == "matched_budget"
    ]
    reference_index = indexed(reference_rows, "reference predictions")
    reference_predictions = [reference_index[key]["prediction"] for key in ids]
    engine = Engine(ids, baseline, records, examples, args.action, args.cost_weight)

    # C4-4: for each held-out outer fold, select weights using only other
    # source folds.  Target-fold records are excluded from inner retrieval.
    tuned_scores, selected_weights = {}, {}
    grid = weight_grid(args.weight_grid_step)
    for target_fold in range(args.folds):
        inner_ids = [key for key in ids if baseline[key]["outer_fold"] != target_fold]
        best = None
        fold_started = time.perf_counter()
        for grid_index, weights in enumerate(grid, 1):
            config_started = time.perf_counter()
            print(
                f"C4-4_TUNING fold={target_fold + 1}/{args.folds} "
                f"config={grid_index}/{len(grid)} weights={weights}",
                flush=True,
            )
            config = dict(REFERENCE, weights=weights)
            scores = {}
            for query_index, key in enumerate(inner_ids, 1):
                scores[key] = engine.query_score(
                    key, config, extra_excluded_folds={target_fold},
                )[0]
                if query_index % 250 == 0 or query_index == len(inner_ids):
                    elapsed = time.perf_counter() - config_started
                    rate = query_index / elapsed if elapsed else 0.0
                    eta = (len(inner_ids) - query_index) / rate if rate else 0.0
                    print(
                        f"C4-4_PROGRESS fold={target_fold + 1}/{args.folds} "
                        f"config={grid_index}/{len(grid)} queries={query_index}/{len(inner_ids)} "
                        f"rate={rate:.2f}/s eta_s={eta:.1f}",
                        flush=True,
                    )
            selected = {key for key in inner_ids if scores[key] > args.utility_threshold}
            gold = [baseline[key]["gold_label"] for key in inner_ids]
            predictions = [
                candidate[key]["prediction"] if key in selected else baseline[key]["prediction"]
                for key in inner_ids
            ]
            objective = (macro_f1(predictions, gold), accuracy(predictions, gold), tuple(weights))
            if best is None or objective > best[0]:
                best = (objective, weights)
        selected_weights[target_fold] = best[1]
        config = dict(REFERENCE, weights=best[1])
        for key in ids:
            if baseline[key]["outer_fold"] == target_fold:
                tuned_scores[key] = engine.query_score(key, config)[0]
        print(
            f"C4-4_FOLD_DONE fold={target_fold + 1}/{args.folds} weights={best[1]} "
            f"elapsed_s={time.perf_counter() - fold_started:.1f}",
            flush=True,
        )

    # C6-4: an explicitly trained source-fold logistic comparator.
    logistic_scores = {}
    for target_fold in range(args.folds):
        print(f"C6-4_TRAIN fold={target_fold + 1}/{args.folds}", flush=True)
        train_ids = [key for key in ids if baseline[key]["outer_fold"] != target_fold]
        target_ids = [key for key in ids if baseline[key]["outer_fold"] == target_fold]
        model = fit_softmax(
            [feature_vector(baseline[key]) for key in train_ids],
            [outcome(baseline[key], candidate[key]) for key in train_ids],
            seed=args.seed + target_fold,
        )
        values = predict_utility(model, [baseline[key] for key in target_ids], args.cost_weight)
        logistic_scores.update({key: float(value) for key, value in zip(target_ids, values)})

    results, rows, routes, caches = {}, [], [], {}
    for name, scores in (
        ("C4-4_source_fold_tuned_weights", tuned_scores),
        ("C6-4_local_logistic_model", logistic_scores),
    ):
        current, current_rows, current_routes, cache = evaluate_and_rows(
            name, ids, baseline, candidate, scores, matched_count, args.utility_threshold,
        )
        results.update(current); rows.extend(current_rows); routes.extend(current_routes); caches[name] = cache

    gold = [baseline[key]["gold_label"] for key in ids]
    comparisons, raw_p = {}, {}
    for name in caches:
        predictions = caches[name]["matched_budget"]
        test = mcnemar_exact(gold, predictions, reference_predictions)
        raw_p[name] = test["exact_two_sided_p"]
        comparisons[name] = {
            "mcnemar_reference_vs_variant": test,
            "paired_bootstrap_reference_minus_variant": paired_bootstrap(
                gold, predictions, reference_predictions, args.bootstrap_samples, args.seed,
            ),
            "fold_analysis": fold_analysis(ids, baseline, reference_predictions, predictions),
        }
    for name, value in holm_adjust(raw_p).items():
        comparisons[name]["holm_adjusted_accuracy_p"] = value
    summary = {
        "evaluation": "nested_source_fold_group_c_p2_comparators",
        "configuration": {
            "matched_route_count": matched_count,
            "matched_route_rate": matched_count / len(ids),
            "weight_grid_step": args.weight_grid_step,
            "weight_selection_objective": "source-inner macro_f1, then accuracy",
            "logistic_features": "B1 label probabilities, predicted-label one-hot, margin, entropy, claim/evidence sizes",
            "logistic_training": "outer-fold-excluded multinomial softmax; outcome classes harmful/neutral/helpful",
            "seed": args.seed,
        },
        "grouping": grouping, "selected_weights_by_outer_fold": selected_weights,
        "results": results, "reference_comparisons": comparisons,
        "scope_warning": "C6-4 is a trained-router comparator and is not part of the training-free method family.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    write_jsonl(args.output_dir / "routes.jsonl", routes)
    print(json.dumps({"output_dir": str(args.output_dir), "results": results}, indent=2))


if __name__ == "__main__":
    main()
