"""Isolated M3b disagreement-aware learned arbitration experiment.

Consumes immutable meta-routing artifacts. It never modifies V3 predictions,
memory, manifests, the frozen method, or the paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, write_jsonl
from errorflow.meta_routing.learned_gate import (
    FEATURE_NAMES, LogisticGate, feature_vector, fit_logistic, subset_vector,
)
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap


L2_GRID = (0.01, 0.1, 1.0, 10.0)
THRESHOLD_GRID = (0.30, 0.40, 0.50, 0.60, 0.70)
EXISTING_POLICIES = (
    "D0_HardLabel", "D1_NLF", "D2_Union", "M1_MaxUtility",
    "M2_Reliability", "M3a_DisagreementGate",
    "M3_ConstantMajority", "M3_ConfidenceControl",
)
ABLATIONS = {
    "M3b_full": tuple(FEATURE_NAMES),
    **{
        f"M3b_without_{name}": tuple(feature for feature in FEATURE_NAMES if feature != name)
        for name in FEATURE_NAMES
    },
}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_meta(directory):
    directory = Path(directory)
    features = {str(row["example_id"]): row for row in read_jsonl(directory / "per_query_features.jsonl")}
    predictions, random_rows = {}, defaultdict(dict)
    for row in read_jsonl(directory / "per_query_predictions.jsonl"):
        if row.get("mode") != "natural":
            continue
        key = str(row["example_id"])
        if row["policy"] == "D4_Random":
            random_rows[int(row["seed"])][key] = row
        else:
            predictions[(row["policy"], key)] = row
    configs = json.loads((directory / "method_configs.json").read_text(encoding="utf-8"))
    ids = sorted(features)
    for policy in ("B1", "A3-All"):
        missing = [key for key in ids if (policy, key) not in predictions]
        if missing:
            raise ValueError(f"Missing {policy} rows: {missing[:3]}")
    return ids, features, predictions, random_rows, configs


def fixed_disagreement(row):
    decisions = row["decisions"]
    return bool(decisions["D0_HardLabel"]["route"]) != bool(decisions["D1_NLF"]["route"])


def outcome(key, predictions):
    b1, a3 = predictions[("B1", key)], predictions[("A3-All", key)]
    gold = b1["gold_label"]
    before = b1["prediction"] == gold
    after = a3["prediction"] == gold
    if before == after:
        return "neutral", None
    return ("helpful", 1) if after else ("harmful", 0)


def source_examples(ids, features, predictions):
    examples = []
    counts = Counter()
    for key in ids:
        row = features[key]
        if not fixed_disagreement(row):
            continue
        status, target = outcome(key, predictions)
        counts[status] += 1
        examples.append({
            "example_id": key, "outer_fold": int(row["outer_fold"]),
            "values": feature_vector(row), "status": status, "target": target,
        })
    counts["disagreements"] = len(examples)
    counts["actionable"] = counts["helpful"] + counts["harmful"]
    for name in ("helpful", "harmful", "neutral"):
        counts[name] += 0
    return examples, dict(counts)


def verify_group_safety(ids, features):
    folds = {key: int(features[key]["outer_fold"]) for key in ids}
    violations = []
    for key in ids:
        query_fold = folds[key]
        for neighborhood in ("hard", "nlf"):
            for neighbor in features[key][neighborhood].get("neighbor_claim_ids", []):
                if neighbor in folds and folds[neighbor] == query_fold:
                    violations.append((key, neighborhood, neighbor, query_fold))
    if violations:
        raise ValueError(f"Grouped-fold memory leakage: {violations[:3]}")
    return {"checked_queries": len(ids), "violations": 0}


def fit_for(examples, feature_names, l2, threshold):
    actionable = [row for row in examples if row["target"] is not None]
    if not actionable:
        raise ValueError("No actionable disagreements available for fitting")
    indexes = tuple(FEATURE_NAMES.index(name) for name in feature_names)
    return fit_logistic(
        [row["values"] for row in actionable], [row["target"] for row in actionable],
        indexes, l2, threshold,
    )


def select_hyperparameters(examples, feature_names, l2_grid, threshold_grid):
    folds = sorted({row["outer_fold"] for row in examples})
    rows = []
    for l2 in l2_grid:
        probabilities = []
        labels = []
        for held_out in folds:
            train = [row for row in examples if row["outer_fold"] != held_out and row["target"] is not None]
            valid = [row for row in examples if row["outer_fold"] == held_out and row["target"] is not None]
            if not train or not valid:
                continue
            model = fit_for(train, feature_names, l2, 0.5)
            probabilities.extend(model.probability(subset_vector(row["values"], feature_names)) for row in valid)
            labels.extend(row["target"] for row in valid)
        for threshold in threshold_grid:
            decisions = [probability >= threshold for probability in probabilities]
            correct = sum(decision == bool(label) for decision, label in zip(decisions, labels))
            rows.append({
                "l2": l2, "threshold": threshold, "n": len(labels),
                "correct": correct, "decision_accuracy": correct / len(labels) if labels else 0.0,
                "route_rate": sum(decisions) / len(decisions) if decisions else 0.0,
            })
    if not rows:
        raise ValueError("Nested selection produced no validation predictions")
    best = max(
        rows,
        key=lambda row: (
            row["decision_accuracy"], -abs(row["threshold"] - 0.5),
            -row["route_rate"], -row["l2"],
        ),
    )
    return best, rows


def train_source_models(examples, feature_names, l2_grid, threshold_grid):
    folds = sorted({row["outer_fold"] for row in examples})
    fold_models, fold_selection = {}, {}
    for outer_fold in folds:
        outer_train = [row for row in examples if row["outer_fold"] != outer_fold]
        selected, trace = select_hyperparameters(
            outer_train, feature_names, l2_grid, threshold_grid
        )
        model = fit_for(
            outer_train, feature_names, selected["l2"], selected["threshold"]
        )
        fold_models[outer_fold] = model
        fold_selection[str(outer_fold)] = {"selected": selected, "inner_cv": trace}
    deploy_selected, deploy_trace = select_hyperparameters(
        examples, feature_names, l2_grid, threshold_grid
    )
    deploy_model = fit_for(
        examples, feature_names, deploy_selected["l2"], deploy_selected["threshold"]
    )
    return fold_models, deploy_model, {
        "outer_fold_selection": fold_selection,
        "deployable_selection": deploy_selected,
        "deployable_group_cv": deploy_trace,
    }


def gate_decisions(ids, features, models_by_fold=None, deploy_model=None):
    routes, probabilities, scores = set(), {}, {}
    for key in ids:
        row = features[key]
        hard = bool(row["decisions"]["D0_HardLabel"]["route"])
        nlf = bool(row["decisions"]["D1_NLF"]["route"])
        if hard == nlf:
            route = hard
            probability = 1.0 if route else 0.0
            score = 1.0 + max(0.0, float(row["decisions"]["D2_Union"]["score"] or 0.0)) if route else -1.0
        else:
            model = deploy_model if deploy_model is not None else models_by_fold[int(row["outer_fold"])]
            values = subset_vector(feature_vector(row), model.feature_names)
            probability = model.probability(values)
            route = probability >= model.threshold
            # Threshold-relative margin keeps fold-specific OOF scores comparable
            # and guarantees that the natural route set is recovered at its own
            # exact matched budget.
            score = probability - model.threshold
        if route:
            routes.add(key)
        probabilities[key], scores[key] = probability, score
    return routes, probabilities, scores


def existing_routes(ids, features, policy):
    return {key for key in ids if bool(features[key]["decisions"][policy]["route"])}


def exact_count(ids, scores, count):
    def finite_score(key):
        value = scores.get(key)
        if value is None:
            return -math.inf
        value = float(value)
        return value if math.isfinite(value) else -math.inf
    ranked = sorted(ids, key=lambda key: (-finite_score(key), key))
    return set(ranked[:count])


def predictions_for(ids, selected, predictions):
    return [
        predictions[("A3-All", key)]["prediction"] if key in selected
        else predictions[("B1", key)]["prediction"]
        for key in ids
    ]


def metrics_for(ids, selected, predictions):
    gold = [predictions[("B1", key)]["gold_label"] for key in ids]
    before = [predictions[("B1", key)]["prediction"] for key in ids]
    after = predictions_for(ids, selected, predictions)
    helpful = sum(b != g and a == g for g, b, a in zip(gold, before, after))
    harmful = sum(b == g and a != g for g, b, a in zip(gold, before, after))
    return {
        "n": len(ids), "routed": len(selected), "route_rate": len(selected) / len(ids),
        "accuracy": accuracy(after, gold), "macro_f1": macro_f1(after, gold),
        "helpful": helpful, "harmful": harmful, "net": helpful - harmful,
        "avg_model_forward_passes": 4.0 + 4.0 * len(selected) / len(ids),
    }, after


def decision_analysis(ids, routes, features, predictions):
    disagreement = [key for key in ids if fixed_disagreement(features[key])]
    actionable = [key for key in disagreement if outcome(key, predictions)[1] is not None]
    correct = sum((key in routes) == bool(outcome(key, predictions)[1]) for key in actionable)
    return {
        "disagreements": len(disagreement), "actionable_disagreements": len(actionable),
        "neutral_disagreements": len(disagreement) - len(actionable),
        "actionable_decision_correct": correct,
        "actionable_decision_accuracy": correct / len(actionable) if actionable else 0.0,
        "oracle_regret": len(actionable) - correct,
    }


def random_routes_from_artifact(ids, random_rows, seed):
    rows = random_rows.get(seed)
    if not rows:
        raise ValueError(f"Random arbitration seed {seed} unavailable")
    return {key for key in ids if bool(rows[key]["routed"])}


def random_budget_routes(ids, seed, count):
    scores = {
        key: int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16)
        for key in ids
    }
    return exact_count(ids, scores, count)


def compare(gold, left, right, samples, seed):
    return {
        "orientation": "right_minus_left",
        "mcnemar_exact": mcnemar_exact(gold, left, right),
        "paired_bootstrap": paired_bootstrap(gold, left, right, samples, seed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-meta-dir", required=True, type=Path)
    parser.add_argument("--target-meta-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["source-oof", "external"], required=True)
    parser.add_argument("--source-model-dir", type=Path)
    parser.add_argument("--l2-grid", nargs="+", type=float, default=list(L2_GRID))
    parser.add_argument("--threshold-grid", nargs="+", type=float, default=list(THRESHOLD_GRID))
    parser.add_argument("--random-seeds", nargs="+", type=int, default=list(range(2026, 2046)))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.mode == "external" and args.source_model_dir is None:
        raise ValueError("External evaluation requires --source-model-dir")
    if args.mode == "source-oof" and args.source_meta_dir.resolve() != args.target_meta_dir.resolve():
        raise ValueError("source-oof requires identical source and target meta directories")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_ids, source_features, source_predictions, _, source_configs = load_meta(args.source_meta_dir)
    source_safety = verify_group_safety(source_ids, source_features)
    examples, training_counts = source_examples(source_ids, source_features, source_predictions)
    target_ids, target_features, target_predictions, target_random, target_configs = load_meta(args.target_meta_dir)

    models, selection, probabilities, gate_scores, variant_routes = {}, {}, {}, {}, {}
    if args.mode == "source-oof":
        for variant, feature_names in ABLATIONS.items():
            fold_models, deploy_model, trace = train_source_models(
                examples, feature_names, args.l2_grid, args.threshold_grid
            )
            routes, probs, scores = gate_decisions(
                target_ids, target_features, models_by_fold=fold_models
            )
            variant_routes[variant], probabilities[variant], gate_scores[variant] = routes, probs, scores
            models[variant] = {
                "fold_models": {str(fold): model.to_dict() for fold, model in fold_models.items()},
                "deployable_model": deploy_model.to_dict(),
            }
            selection[variant] = trace
    else:
        bundle = json.loads((args.source_model_dir / "models.json").read_text(encoding="utf-8"))
        source_backbone = bundle.get("source_config", {}).get("backbone_id")
        target_backbone = target_configs.get("backbone_id")
        if source_backbone and target_backbone and source_backbone != target_backbone:
            raise ValueError(
                f"Backbone mismatch: source model={source_backbone} target={target_backbone}"
            )
        for variant, payload in bundle["variants"].items():
            deploy_model = LogisticGate.from_dict(payload["deployable_model"])
            routes, probs, scores = gate_decisions(
                target_ids, target_features, deploy_model=deploy_model
            )
            variant_routes[variant], probabilities[variant], gate_scores[variant] = routes, probs, scores
        models = bundle["variants"]
        selection = bundle["selection"]

    routes = {policy: existing_routes(target_ids, target_features, policy) for policy in EXISTING_POLICIES}
    routes.update(variant_routes)
    primary_seed = args.seed if args.seed in target_random else sorted(target_random)[0]
    routes[f"D4_Random_seed{primary_seed}"] = random_routes_from_artifact(
        target_ids, target_random, primary_seed
    )

    natural_rows, natural_predictions, analyses, prediction_rows = [], {}, {}, []
    for policy, selected in routes.items():
        metrics, predicted = metrics_for(target_ids, selected, target_predictions)
        analysis = decision_analysis(target_ids, selected, target_features, target_predictions)
        analyses[policy] = analysis
        natural_predictions[policy] = predicted
        natural_rows.append({"mode": "natural", "policy": policy, **metrics, **analysis})
        for key, value in zip(target_ids, predicted):
            prediction_rows.append({
                "example_id": key, "mode": "natural", "policy": policy,
                "routed": key in selected, "prediction": value,
                "gold_label": target_predictions[("B1", key)]["gold_label"],
                "gate_probability": probabilities.get(policy, {}).get(key),
            })

    random_natural = []
    for seed in args.random_seeds:
        if seed not in target_random:
            continue
        selected = random_routes_from_artifact(target_ids, target_random, seed)
        metrics, predicted = metrics_for(target_ids, selected, target_predictions)
        seed_row = {"seed": seed, **metrics, **decision_analysis(
            target_ids, selected, target_features, target_predictions
        )}
        random_natural.append(seed_row)
        natural_rows.append({"mode": "natural", "policy": "D4_Random", **seed_row})
        for key, value in zip(target_ids, predicted):
            prediction_rows.append({
                "example_id": key, "mode": "natural", "policy": "D4_Random",
                "seed": seed, "routed": key in selected, "prediction": value,
                "gold_label": target_predictions[("B1", key)]["gold_label"],
            })
    if random_natural:
        natural_rows.append({
            "mode": "natural", "policy": "D4_Random_mean",
            **{
                name: statistics.mean(row[name] for row in random_natural)
                for name in ("routed", "route_rate", "accuracy", "macro_f1", "helpful", "harmful", "net",
                             "avg_model_forward_passes", "actionable_decision_accuracy", "oracle_regret")
            },
            "accuracy_std": statistics.pstdev(row["accuracy"] for row in random_natural),
            "macro_f1_std": statistics.pstdev(row["macro_f1"] for row in random_natural),
        })

    matched_count = len(routes["M3b_full"])
    matched_routes = {}
    for policy in EXISTING_POLICIES:
        scores = {key: target_features[key]["decisions"][policy]["score"] for key in target_ids}
        matched_routes[policy] = exact_count(target_ids, scores, matched_count)
    for variant in ABLATIONS:
        matched_routes[variant] = exact_count(target_ids, gate_scores[variant], matched_count)
    if matched_routes["M3b_full"] != routes["M3b_full"]:
        raise ValueError("M3b natural route set was not reproduced at its exact matched budget")
    matched_routes[f"Random_Budget_seed{primary_seed}"] = random_budget_routes(
        target_ids, primary_seed, matched_count
    )

    matched_rows, matched_predictions = [], {}
    for policy, selected in matched_routes.items():
        metrics, predicted = metrics_for(target_ids, selected, target_predictions)
        analysis = decision_analysis(target_ids, selected, target_features, target_predictions)
        matched_predictions[policy] = predicted
        matched_rows.append({
            "mode": "matched_m3b_budget", "matched_count": matched_count,
            "policy": policy, **metrics, **analysis,
        })
        for key, value in zip(target_ids, predicted):
            prediction_rows.append({
                "example_id": key, "mode": "matched_m3b_budget", "policy": policy,
                "routed": key in selected, "prediction": value,
                "gold_label": target_predictions[("B1", key)]["gold_label"],
                "gate_probability": probabilities.get(policy, {}).get(key),
            })

    random_matched = []
    for seed in args.random_seeds:
        selected = random_budget_routes(target_ids, seed, matched_count)
        metrics, predicted = metrics_for(target_ids, selected, target_predictions)
        analysis = decision_analysis(target_ids, selected, target_features, target_predictions)
        random_matched.append({"seed": seed, **metrics, **analysis})
        if seed != primary_seed:
            matched_rows.append({
                "mode": "matched_m3b_budget", "matched_count": matched_count,
                "policy": "Random-Budget-Control", "seed": seed, **metrics, **analysis,
            })
            for key, value in zip(target_ids, predicted):
                prediction_rows.append({
                    "example_id": key, "mode": "matched_m3b_budget",
                    "policy": "Random-Budget-Control", "seed": seed,
                    "routed": key in selected, "prediction": value,
                    "gold_label": target_predictions[("B1", key)]["gold_label"],
                })
    if random_matched:
        matched_rows.append({
            "mode": "matched_m3b_budget", "matched_count": matched_count,
            "policy": "Random-Budget-Control_mean",
            **{
                name: statistics.mean(row[name] for row in random_matched)
                for name in ("routed", "route_rate", "accuracy", "macro_f1", "helpful", "harmful", "net",
                             "avg_model_forward_passes", "actionable_decision_accuracy", "oracle_regret")
            },
            "accuracy_std": statistics.pstdev(row["accuracy"] for row in random_matched),
            "macro_f1_std": statistics.pstdev(row["macro_f1"] for row in random_matched),
        })

    gold = [target_predictions[("B1", key)]["gold_label"] for key in target_ids]
    natural_comparisons, matched_comparisons = {}, {}
    for policy, predicted in natural_predictions.items():
        if policy == "M3b_full":
            continue
        natural_comparisons[f"M3b_full_vs_{policy}"] = compare(
            gold, predicted, natural_predictions["M3b_full"],
            args.bootstrap_samples, args.seed,
        )
    for policy, predicted in matched_predictions.items():
        if policy == "M3b_full":
            continue
        matched_comparisons[f"M3b_full_vs_{policy}"] = compare(
            gold, predicted, matched_predictions["M3b_full"],
            args.bootstrap_samples, args.seed,
        )

    per_disagreement = []
    for key in target_ids:
        if not fixed_disagreement(target_features[key]):
            continue
        status, target = outcome(key, target_predictions)
        row = {
            "example_id": key, "outcome": status, "actionable_target": target,
            "m3b_probability": probabilities["M3b_full"][key],
        }
        for policy in EXISTING_POLICIES + ("M3b_full", f"D4_Random_seed{primary_seed}"):
            route = key in routes[policy]
            row[f"{policy}_route"] = route
            row[f"{policy}_oracle_regret"] = int(route != bool(target)) if target is not None else 0
        per_disagreement.append(row)

    model_bundle = {
        "evaluation": "M3b_disagreement_aware_l2_logistic",
        "source_meta_dir": str(args.source_meta_dir),
        "features": list(FEATURE_NAMES), "l2_grid": args.l2_grid,
        "threshold_grid": args.threshold_grid, "training_counts": training_counts,
        "group_safety": source_safety, "variants": models, "selection": selection,
        "neutral_policy": "excluded_from_binary_fitting_and_actionable_accuracy; retained for routing evaluation",
        "nonfinite_utility_encoding": "zero; eligibility remains implicit in the disagreement-only gate scope",
        "source_config": source_configs,
    }
    if args.mode == "source-oof":
        atomic_write_json(args.output_dir / "models.json", model_bundle)
    audit = {
        "isolated_output": True,
        "source_only_fitting": True,
        "target_labels_used_for_fitting_or_selection": False,
        "cached_predictions_modified": False,
        "frozen_manifests_modified": False,
        "feature_count": len(FEATURE_NAMES),
        "features": list(FEATURE_NAMES),
        "source_group_safety": source_safety,
        "training_counts": training_counts,
        "preserved_findings": {
            "M2": "grouped source evaluation may use cross-fitted fold-specific beta values",
            "M3a": "may select different neighborhoods while reproducing Union final actions",
        },
        "M3b_fit_mode": "nested_group_cross_validation" if args.mode == "source-oof" else "frozen_source_model",
    }
    summary = {
        "evaluation": "M3b_disagreement_aware_learned_arbitration",
        "mode": args.mode, "matched_route_count": matched_count,
        "natural": natural_rows, "matched_budget": matched_rows,
        "decision_analysis": analyses,
        "feature_ablation_variants": list(ABLATIONS),
        "recommendation_rule": "retain ERRORFlow V3 unchanged unless M3b consistently exceeds the best fixed policy",
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    atomic_write_json(args.output_dir / "statistical_comparisons.json", {
        "natural": natural_comparisons, "matched_budget": matched_comparisons,
    })
    atomic_write_json(args.output_dir / "audit.json", audit)
    write_csv(args.output_dir / "natural_results.csv", natural_rows)
    write_csv(args.output_dir / "matched_budget_results.csv", matched_rows)
    write_csv(args.output_dir / "oracle_regret.csv", per_disagreement)
    write_jsonl(args.output_dir / "per_query_predictions.jsonl", prediction_rows)
    ablation_rows = [row for row in natural_rows if row["policy"] in ABLATIONS]
    write_csv(args.output_dir / "feature_ablation.csv", ablation_rows)

    fixed = [row for row in natural_rows if row["policy"] in ("D0_HardLabel", "D1_NLF", "D2_Union")]
    full = next(row for row in natural_rows if row["policy"] == "M3b_full")
    best_fixed = max(fixed, key=lambda row: (row["accuracy"], row["macro_f1"]))
    consistent = full["accuracy"] > best_fixed["accuracy"] and full["macro_f1"] > best_fixed["macro_f1"]
    recommendation = (
        "M3b exceeds the best fixed policy descriptively on both metrics; require cross-setting consistency and paired tests."
        if consistent else
        "M3b does not beat the best fixed policy on both Accuracy and Macro-F1; retain ERRORFlow V3 unchanged."
    )
    report = [
        "# M3b disagreement-aware learned arbitration", "",
        f"- Mode: `{args.mode}`", f"- Source actionable training cases: `{training_counts['actionable']}`",
        f"- Source neutral disagreements: `{training_counts['neutral']}`",
        f"- Natural M3b routes: `{full['routed']}`",
        f"- M3b Accuracy / Macro-F1: `{full['accuracy']:.6f}` / `{full['macro_f1']:.6f}`",
        f"- Best fixed policy: `{best_fixed['policy']}` (`{best_fixed['accuracy']:.6f}`, `{best_fixed['macro_f1']:.6f}`)",
        f"- Recommendation: **{recommendation}**", "",
        "M2 fold-specific beta behavior and M3a/Union action equivalence are preserved as audited semantics, not treated as bugs.",
    ]
    (args.output_dir / "experiment_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "m3b": full, "best_fixed": best_fixed,
                      "recommendation": recommendation}, indent=2))
    print(f"M3B_DONE rows={len(target_ids)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
