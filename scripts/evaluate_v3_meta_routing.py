"""Exploratory M1/M2/M3 arbitration over frozen V3 B1/A3 caches.

The script materializes all routes before consulting gold labels.  It never
changes the supplied predictions, memory, manifests, or existing V3 outputs.
M3b is intentionally omitted: this dependency-free implementation evaluates
the strictly training-free M3a gate and records M3b as not run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.error_memory import TOKEN_RE
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.meta_routing import (
    CANONICAL_TOKENIZER, PolicyDecision, build_meta_features, decide_policies,
)
from errorflow.v3_memory import ResidualMemoryStore, record_from_dict
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap, percentile

try:
    from scripts.evaluate_v3_grouping_sensitivity import grouped_folds, near_duplicate_edges
except ModuleNotFoundError:
    from evaluate_v3_grouping_sensitivity import grouped_folds, near_duplicate_edges


POLICIES = (
    "D0_HardLabel", "D1_NLF", "D2_Union", "D3_Intersection",
    "M1_MaxUtility", "M2_Reliability", "M3a_DisagreementGate",
    "M3_ConstantMajority", "M3_ConfidenceControl",
)
BETAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gold_label(row):
    value = row.get("label", row.get("gold_label"))
    if value is None:
        raise ValueError(f"Missing gold label: {row.get('example_id')}")
    return value


def per_class(gold, predicted):
    result = {}
    for label in sorted(set(gold) | set(predicted)):
        tp = sum(g == p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": sum(g == label for g in gold),
        }
    return result


def evaluate(ids, gold_rows, baseline, candidate, selected):
    gold = [gold_label(gold_rows[key]) for key in ids]
    before = [baseline[key]["prediction"] for key in ids]
    after = [candidate[key]["prediction"] if key in selected else baseline[key]["prediction"] for key in ids]
    helpful = sum(b != g and a == g for g, b, a in zip(gold, before, after))
    harmful = sum(b == g and a != g for g, b, a in zip(gold, before, after))
    return {
        "n": len(ids), "routed": len(selected), "route_rate": len(selected) / len(ids),
        "accuracy": accuracy(after, gold), "macro_f1": macro_f1(after, gold),
        "helpful": helpful, "harmful": harmful, "net": helpful - harmful,
        "avg_model_forward_passes": 4.0 + 4.0 * len(selected) / len(ids),
        "per_class": per_class(gold, after),
    }, after


def exact_count(ids, scores, count):
    if not 0 <= count <= len(ids):
        raise ValueError("Invalid requested route count")
    ranked = sorted(ids, key=lambda key: (-scores.get(key, -math.inf), key))
    return set(ranked[:count])


def holm_adjust(values):
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    result, running = {}, 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (len(ordered) - index)))
        result[name] = running
    return result


def cluster_bootstrap(gold, left, right, groups, samples, seed):
    rng = random.Random(seed)
    members = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    keys = sorted(members, key=str)
    acc, f1 = [], []
    for _ in range(samples):
        chosen = [rng.choice(keys) for _ in keys]
        indexes = [index for group in chosen for index in members[group]]
        g = [gold[index] for index in indexes]
        l = [left[index] for index in indexes]
        r = [right[index] for index in indexes]
        acc.append(accuracy(r, g) - accuracy(l, g))
        f1.append(macro_f1(r, g) - macro_f1(l, g))
    return {
        "samples": samples, "seed": seed,
        "accuracy_delta_ci95": [percentile(acc, 0.025), percentile(acc, 0.975)],
        "macro_f1_delta_ci95": [percentile(f1, 0.025), percentile(f1, 0.975)],
    }


def sibling_manifest(path):
    path = Path(path)
    candidate = path.with_suffix(".manifest.json")
    return json.loads(candidate.read_text(encoding="utf-8")) if candidate.exists() else None


def validate_memory_outcomes(records, source_baseline, source_candidate):
    if source_baseline is None or source_candidate is None:
        return {"status": "not_checked", "reason": "source caches not supplied"}
    baseline = indexed(read_jsonl(source_baseline), "memory baseline")
    candidate = indexed(read_jsonl(source_candidate), "memory candidate")
    mismatches = []
    for record in records:
        key = record.source_claim_id
        if key not in baseline or key not in candidate:
            mismatches.append(f"missing:{key}")
            continue
        row, after = baseline[key], candidate[key]
        gold = row.get("gold_label", row.get("label"))
        before_correct = row["prediction"] == gold
        after_correct = after["prediction"] == gold
        expected = (int(not before_correct and after_correct), int(before_correct and not after_correct))
        actual = (record.recovery_count, record.regression_count)
        if expected != actual:
            mismatches.append(f"outcome:{key}:{expected}!={actual}")
    return {"status": "ok" if not mismatches else "mismatch", "checked": len(records), "mismatches": mismatches[:50]}


def prepare_grouped(baseline_rows, records, threshold, folds, seed, assignment_path=None):
    if assignment_path:
        payload = json.loads(Path(assignment_path).read_text(encoding="utf-8"))
        assignments = {str(key): int(value) for key, value in payload.items()}
        components = [[key] for key in assignments]
    else:
        edges = near_duplicate_edges(baseline_rows, threshold)
        assignments, components = grouped_folds(baseline_rows, edges, threshold, folds, seed)
    updated_rows = []
    for source in baseline_rows:
        row = dict(source)
        row["outer_fold"] = assignments[str(row["example_id"])]
        updated_rows.append(row)
    updated_records = []
    for record in records:
        provenance = dict(record.provenance)
        provenance["outer_fold"] = assignments[record.source_claim_id]
        updated_records.append(replace(record, provenance=provenance))
    component_by_id = {}
    for index, component in enumerate(components):
        for key in component:
            component_by_id[key] = index
    return updated_rows, updated_records, assignments, component_by_id


def metric_rank(metrics):
    return (metrics["macro_f1"], metrics["accuracy"], -metrics["route_rate"])


def safe_mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def source_beta_selection(ids, folds_by_id, features, gold_rows, baseline, candidate, betas):
    """Cross-fitted source-only beta selection; no target-domain labels."""
    selected_by_fold, validation = {}, {}
    unique_folds = sorted(set(folds_by_id.values()))
    for held_out in unique_folds:
        train_ids = [key for key in ids if folds_by_id[key] != held_out]
        scores = {}
        for beta in betas:
            chosen = {
                key for key in train_ids
                if decide_policies(features[key], beta=beta)["M2_Reliability"].route
            }
            metrics, _ = evaluate(train_ids, gold_rows, baseline, candidate, chosen)
            scores[beta] = metrics
        selected_by_fold[held_out] = max(betas, key=lambda beta: (metric_rank(scores[beta]), -beta))
        validation[str(held_out)] = {str(beta): scores[beta] for beta in betas}
    all_scores = {}
    for beta in betas:
        chosen = {key for key in ids if decide_policies(features[key], beta=beta)["M2_Reliability"].route}
        all_scores[beta], _ = evaluate(ids, gold_rows, baseline, candidate, chosen)
    deployable = max(betas, key=lambda beta: (metric_rank(all_scores[beta]), -beta))
    return selected_by_fold, deployable, {"fold_validation": validation, "aggregate": {str(k): v for k, v in all_scores.items()}}


def oracle_analysis(ids, gold_rows, baseline, candidate, hard_routes, nlf_routes):
    agreement = {key for key in ids if (key in hard_routes) == (key in nlf_routes)}
    disagreement = set(ids) - agreement
    oracle_routes = set(hard_routes & nlf_routes)
    affected = hard_only_help = hard_only_harm = nlf_only_help = nlf_only_harm = 0
    for key in disagreement:
        gold = gold_label(gold_rows[key])
        before, after = baseline[key]["prediction"], candidate[key]["prediction"]
        before_correct, after_correct = before == gold, after == gold
        affected += before_correct != after_correct
        if after_correct and not before_correct:
            oracle_routes.add(key)
        elif before_correct and not after_correct:
            oracle_routes.discard(key)
        elif key in nlf_routes:  # deterministic tie preserving NLF.
            oracle_routes.add(key)
        if key in hard_routes and key not in nlf_routes:
            hard_only_help += (not before_correct and after_correct)
            hard_only_harm += (before_correct and not after_correct)
        if key in nlf_routes and key not in hard_routes:
            nlf_only_help += (not before_correct and after_correct)
            nlf_only_harm += (before_correct and not after_correct)
    metrics, predictions = evaluate(ids, gold_rows, baseline, candidate, oracle_routes)
    labels = sorted({gold_label(gold_rows[key]) for key in ids})
    class_upper = {}
    for label in labels:
        support = sum(gold_label(gold_rows[key]) == label for key in ids)
        tp = sum(
            gold_label(gold_rows[key]) == label and
            label in {baseline[key]["prediction"], candidate[key]["prediction"]}
            for key in ids
        )
        forced_fp = sum(
            gold_label(gold_rows[key]) != label and
            baseline[key]["prediction"] == candidate[key]["prediction"] == label
            for key in ids
        )
        fn = support - tp
        class_upper[label] = 2 * tp / (2 * tp + forced_fp + fn) if 2 * tp + forced_fp + fn else 0.0
    return {
        "agreement": len(agreement), "disagreement": len(disagreement),
        "correctness_affecting_disagreements": affected,
        "hard_only": {"helpful": hard_only_help, "harmful": hard_only_harm, "net": hard_only_help-hard_only_harm},
        "nlf_only": {"helpful": nlf_only_help, "harmful": nlf_only_harm, "net": nlf_only_help-nlf_only_harm},
        "accuracy_oracle": metrics,
        "accuracy_oracle_predictions": predictions,
        "macro_f1_diagnostic": {
            "achievable_accuracy_oracle_macro_f1": metrics["macro_f1"],
            "relaxed_classwise_upper_bound": sum(class_upper.values()) / len(class_upper),
            "per_class_relaxed_upper_bound": class_upper,
            "caveat": "Classwise relaxation may choose incompatible actions across labels; it is a ceiling, not a deployable policy.",
        },
    }


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backbone-id", required=True)
    parser.add_argument("--dataset-protocol", required=True)
    parser.add_argument("--protocol", choices=["grouped_internal", "external"], required=True)
    parser.add_argument("--fold-assignments", type=Path)
    parser.add_argument("--memory-baseline", type=Path)
    parser.add_argument("--memory-candidate", type=Path)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--project-conflicting-to-nei", action="store_true")
    parser.add_argument("--action", default="conflicting_rescue_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--group-threshold", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--m2-beta", type=float)
    parser.add_argument("--majority-action", choices=["route", "skip"])
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--beta-grid", nargs="+", type=float, default=list(BETAS))
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--random-seeds", nargs="+", type=int, default=list(range(2026, 2046)))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.top_k < args.min_support:
        raise ValueError("top-k must be at least min-support")
    if args.protocol == "external" and args.m2_beta is None:
        raise ValueError("External evaluation requires source-selected --m2-beta")
    if args.protocol == "external" and (
        args.majority_action is None or args.confidence_threshold is None
    ):
        raise ValueError(
            "External evaluation requires source-selected --majority-action and --confidence-threshold"
        )
    if any(not 0 <= budget <= 1 for budget in args.budgets):
        raise ValueError("budgets must be within [0,1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "gold": args.gold, "baseline": args.baseline,
        "candidate": args.candidate, "memory": args.memory,
    }
    input_hashes_before = {name: sha256_file(path) for name, path in input_paths.items()}
    gold_rows = indexed(read_jsonl(args.gold), "gold")
    baseline_rows = list(read_jsonl(args.baseline))
    candidate = indexed(read_jsonl(args.candidate), "candidate")
    records = [record_from_dict(row) for row in read_jsonl(args.memory)]
    assignments = component_by_id = None
    if args.protocol == "grouped_internal":
        baseline_rows, records, assignments, component_by_id = prepare_grouped(
            baseline_rows, records, args.group_threshold, args.folds, args.seed, args.fold_assignments
        )
    baseline = indexed(baseline_rows, "baseline")
    if set(gold_rows) != set(baseline) or set(gold_rows) != set(candidate):
        raise ValueError("Gold, B1, and A3 IDs must match exactly")
    ids = sorted(gold_rows)
    store = ResidualMemoryStore(records)
    projection = {"CONFLICTING": "NOT ENOUGH INFO"} if args.project_conflicting_to_nei else {}
    eval_baseline = {
        key: {**row, "prediction": projection.get(row["prediction"], row["prediction"])}
        for key, row in baseline.items()
    }
    eval_candidate = {
        key: {**row, "prediction": projection.get(row["prediction"], row["prediction"])}
        for key, row in candidate.items()
    }

    baseline_manifest, candidate_manifest = sibling_manifest(args.baseline), sibling_manifest(args.candidate)
    model_values = {
        model for manifest in (baseline_manifest, candidate_manifest)
        if manifest
        for model in [manifest.get("config", {}).get("model")]
        if model
    }
    if len(model_values) > 1:
        raise ValueError(f"B1/A3 model mismatch: {sorted(model_values)}")
    memory_candidate_manifest = sibling_manifest(args.memory_candidate) if args.memory_candidate else None
    memory_model = (
        memory_candidate_manifest.get("config", {}).get("model")
        if memory_candidate_manifest else None
    )
    if model_values and memory_model and memory_model not in model_values:
        raise ValueError(
            f"Target cache and source memory candidate use different models: "
            f"target={sorted(model_values)} source={memory_model}"
        )
    memory_check = validate_memory_outcomes(records, args.memory_baseline, args.memory_candidate)
    if memory_check["status"] == "mismatch":
        raise ValueError(f"Memory outcomes do not replay: {memory_check['mismatches'][:3]}")
    memory_ids = [record.memory_id for record in records]
    source_ids = [record.source_claim_id for record in records]
    external_overlap = len(set(ids) & set(source_ids)) if args.protocol == "external" else None
    if external_overlap:
        raise ValueError(f"External target/source memory ID overlap: {external_overlap}")

    feature_rows, features, latencies = [], {}, []
    for index, key in enumerate(ids, 1):
        row = baseline[key]
        excluded_folds = {row["outer_fold"]} if args.protocol == "grouped_internal" else set()
        started = time.perf_counter()
        feature = build_meta_features(
            key, row, store, action=args.action, top_k=args.top_k,
            min_support=args.min_support, cost_weight=args.cost_weight,
            exclude_claim_ids={key}, exclude_folds=excluded_folds,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        features[key], latencies = feature, latencies + [elapsed]
        payload = feature.to_dict()
        payload["feature_latency_ms"] = elapsed
        payload["outer_fold"] = row.get("outer_fold")
        feature_rows.append(json_safe(payload))
        if index % 250 == 0 or index == len(ids):
            print(f"META_FEATURES progress={index}/{len(ids)}", flush=True)

    if args.protocol == "grouped_internal":
        beta_by_fold, deployable_beta, selection = source_beta_selection(
            ids, assignments, features, gold_rows, eval_baseline, eval_candidate, args.beta_grid
        )
    else:
        beta_by_fold, deployable_beta, selection = None, args.m2_beta, {
            "source_selected_beta": args.m2_beta,
            "selection_performed_on_target": False,
        }

    fixed_preview = {key: decide_policies(features[key], beta=0.0) for key in ids}
    fixed_disagreement = {
        key for key in ids
        if fixed_preview[key]["D0_HardLabel"].route != fixed_preview[key]["D1_NLF"].route
    }
    if args.protocol == "grouped_internal":
        majority_by_fold, confidence_by_fold = {}, {}
        for held_out in sorted(set(assignments.values())):
            training = [
                key for key in fixed_disagreement if assignments[key] != held_out
            ]
            route_wins = sum(
                eval_candidate[key]["prediction"] == gold_label(gold_rows[key])
                and eval_baseline[key]["prediction"] != gold_label(gold_rows[key])
                for key in training
            )
            skip_wins = sum(
                eval_baseline[key]["prediction"] == gold_label(gold_rows[key])
                and eval_candidate[key]["prediction"] != gold_label(gold_rows[key])
                for key in training
            )
            majority_by_fold[held_out] = "route" if route_wins > skip_wins else "skip"
            entropy_values = sorted(features[key].baseline_entropy for key in training)
            confidence_by_fold[held_out] = percentile(entropy_values, 0.5) if entropy_values else math.inf
        all_route_wins = sum(
            eval_candidate[key]["prediction"] == gold_label(gold_rows[key])
            and eval_baseline[key]["prediction"] != gold_label(gold_rows[key])
            for key in fixed_disagreement
        )
        all_skip_wins = sum(
            eval_baseline[key]["prediction"] == gold_label(gold_rows[key])
            and eval_candidate[key]["prediction"] != gold_label(gold_rows[key])
            for key in fixed_disagreement
        )
        deployable_majority = "route" if all_route_wins > all_skip_wins else "skip"
        disagreement_entropy = sorted(features[key].baseline_entropy for key in fixed_disagreement)
        deployable_confidence = percentile(disagreement_entropy, 0.5) if disagreement_entropy else math.inf
    else:
        majority_by_fold = confidence_by_fold = None
        deployable_majority = args.majority_action
        deployable_confidence = args.confidence_threshold

    decisions, routes = {}, {policy: set() for policy in POLICIES}
    applied_beta_by_id = {}
    random_routes = {seed: set() for seed in args.random_seeds}
    for key in ids:
        beta = beta_by_fold[assignments[key]] if beta_by_fold is not None else deployable_beta
        applied_beta_by_id[key] = beta
        row_decisions = decide_policies(features[key], beta=beta, threshold=args.utility_threshold, random_seed=args.seed)
        hard_route = row_decisions["D0_HardLabel"].route
        nlf_route = row_decisions["D1_NLF"].route
        if hard_route == nlf_route:
            majority_route = confidence_route = hard_route
            control_reason = "fixed_policies_agree"
        else:
            fold = assignments[key] if assignments is not None else None
            majority = majority_by_fold[fold] if majority_by_fold is not None else deployable_majority
            confidence_threshold = (
                confidence_by_fold[fold] if confidence_by_fold is not None else deployable_confidence
            )
            majority_route = majority == "route"
            confidence_route = features[key].baseline_entropy >= confidence_threshold
            control_reason = "source_only_disagreement_control"
        row_decisions["M3_ConstantMajority"] = PolicyDecision(
            "M3_ConstantMajority", majority_route,
            1.0 if majority_route else -1.0, None, control_reason,
        )
        row_decisions["M3_ConfidenceControl"] = PolicyDecision(
            "M3_ConfidenceControl", confidence_route,
            features[key].baseline_entropy, None, control_reason,
        )
        decisions[key] = row_decisions
        for policy in POLICIES:
            if row_decisions[policy].route:
                routes[policy].add(key)
        for seed in args.random_seeds:
            if decide_policies(features[key], beta=beta, threshold=args.utility_threshold, random_seed=seed)["D4_Random"].route:
                random_routes[seed].add(key)
    for payload in feature_rows:
        key = payload["example_id"]
        payload["decisions"] = {
            policy: json_safe(decision.to_dict()) for policy, decision in decisions[key].items()
            if policy != "D4_Random"
        }
        payload["applied_m2_beta"] = applied_beta_by_id[key]
        payload["meta_disagreement_flags"] = {
            policy: {
                "vs_hard": decision.route != decisions[key]["D0_HardLabel"].route,
                "vs_nlf": decision.route != decisions[key]["D1_NLF"].route,
            }
            for policy, decision in decisions[key].items()
            if policy.startswith("M")
        }

    if routes["D2_Union"] != routes["M1_MaxUtility"]:
        raise ValueError("M1/Union mismatch under canonical eligibility; audit before proceeding")
    beta0_mismatches = []
    for key in ids:
        beta0 = decide_policies(features[key], beta=0.0, threshold=args.utility_threshold)
        if beta0["M2_Reliability"] != replace(
            beta0["M1_MaxUtility"], policy="M2_Reliability",
            reason="maximum_reliability_adjusted_utility",
        ):
            beta0_mismatches.append(key)
    if beta0_mismatches:
        raise ValueError(f"M2(beta=0) != M1 for {beta0_mismatches[:3]}")

    applied_beta_counts = Counter(str(applied_beta_by_id[key]) for key in ids)
    actual_m2_m1_route_mismatches = [
        key for key in ids
        if decisions[key]["M2_Reliability"].route != decisions[key]["M1_MaxUtility"].route
    ]
    zero_beta_ids = [key for key in ids if abs(applied_beta_by_id[key]) < 1e-15]
    zero_beta_actual_mismatches = [
        key for key in zero_beta_ids
        if decisions[key]["M2_Reliability"].route != decisions[key]["M1_MaxUtility"].route
    ]
    if zero_beta_actual_mismatches:
        raise ValueError(
            "Applied beta=0 produced M2/M1 route mismatch: "
            f"{zero_beta_actual_mismatches[:3]}"
        )

    natural_metrics, natural_predictions = {}, {}
    result_rows, prediction_rows = [], []
    b1_metrics, b1_predictions = evaluate(ids, gold_rows, eval_baseline, eval_candidate, set())
    a3_metrics, a3_predictions = evaluate(ids, gold_rows, eval_baseline, eval_candidate, set(ids))
    for policy, selected in {"B1": set(), "A3-All": set(ids), **routes}.items():
        metrics, predictions = evaluate(ids, gold_rows, eval_baseline, eval_candidate, selected)
        disagreement_ids = sorted(fixed_disagreement)
        disagreement_selected = selected & fixed_disagreement
        disagreement_metrics, _ = evaluate(
            disagreement_ids, gold_rows, eval_baseline, eval_candidate,
            disagreement_selected,
        ) if disagreement_ids else ({"accuracy": 0.0}, [])
        affected = [
            key for key in disagreement_ids
            if (eval_baseline[key]["prediction"] == gold_label(gold_rows[key]))
            != (eval_candidate[key]["prediction"] == gold_label(gold_rows[key]))
        ]
        action_correct = sum(
            ((key in selected) == (eval_candidate[key]["prediction"] == gold_label(gold_rows[key])))
            for key in affected
        )
        metrics["disagreement_n"] = len(disagreement_ids)
        metrics["disagreement_accuracy"] = disagreement_metrics["accuracy"]
        metrics["disagreement_action_correctness"] = (
            action_correct / len(affected) if affected else 0.0
        )
        natural_metrics[policy], natural_predictions[policy] = metrics, predictions
        result_rows.append({"mode": "natural", "policy": policy, **{k: v for k, v in metrics.items() if k != "per_class"}})
        for key, prediction in zip(ids, predictions):
            prediction_rows.append({
                "example_id": key, "mode": "natural", "policy": policy,
                "routed": key in selected, "prediction": prediction,
                "gold_label": gold_label(gold_rows[key]),
            })
    random_metrics, random_predictions_by_seed = [], {}
    for seed, selected in random_routes.items():
        metrics, predictions = evaluate(ids, gold_rows, eval_baseline, eval_candidate, selected)
        random_metrics.append({"seed": seed, **metrics})
        random_predictions_by_seed[seed] = predictions
        result_rows.append({
            "mode": "natural", "policy": "D4_Random", "seed": seed,
            **{key: value for key, value in metrics.items() if key != "per_class"},
        })
        for key, prediction in zip(ids, predictions):
            prediction_rows.append({
                "example_id": key, "mode": "natural", "policy": "D4_Random",
                "seed": seed, "routed": key in selected, "prediction": prediction,
                "gold_label": gold_label(gold_rows[key]),
            })
    result_rows.append({
        "mode": "natural", "policy": "D4_Random_mean",
        **{name: statistics.mean(row[name] for row in random_metrics)
           for name in ("routed", "route_rate", "accuracy", "macro_f1", "helpful", "harmful", "net", "avg_model_forward_passes")},
        "accuracy_std": statistics.pstdev(row["accuracy"] for row in random_metrics),
        "macro_f1_std": statistics.pstdev(row["macro_f1"] for row in random_metrics),
    })

    hard_count, nlf_count = len(routes["D0_HardLabel"]), len(routes["D1_NLF"])
    budgets = sorted(set(args.budgets + [hard_count / len(ids), nlf_count / len(ids)]))
    matched_rows, quality_rows = [], []
    for budget in budgets:
        count = round(len(ids) * budget)
        for policy in POLICIES:
            scores = {key: decisions[key][policy].score for key in ids}
            selected = exact_count(ids, scores, count)
            if len(selected) != count:
                raise AssertionError("Matched budget did not select exact count")
            metrics, predictions = evaluate(ids, gold_rows, eval_baseline, eval_candidate, selected)
            row = {"mode": "matched_budget", "requested_budget": budget, "requested_count": count,
                   "policy": policy, **{k: v for k, v in metrics.items() if k != "per_class"}}
            matched_rows.append(row)
            quality_rows.append(row)
            for key, prediction in zip(ids, predictions):
                prediction_rows.append({
                    "example_id": key, "mode": "matched_budget", "requested_budget": budget,
                    "policy": policy, "routed": key in selected, "prediction": prediction,
                    "gold_label": gold_label(gold_rows[key]),
                })
        random_budget_metrics = []
        for seed in args.random_seeds:
            scores = {
                key: int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16)
                for key in ids
            }
            selected = exact_count(ids, scores, count)
            metrics, predictions = evaluate(ids, gold_rows, eval_baseline, eval_candidate, selected)
            random_budget_metrics.append(metrics)
            seed_row = {
                "mode": "matched_budget", "requested_budget": budget,
                "requested_count": count, "policy": "Random-Budget-Control", "seed": seed,
                **{key: value for key, value in metrics.items() if key != "per_class"},
            }
            matched_rows.append(seed_row)
            for key, prediction in zip(ids, predictions):
                prediction_rows.append({
                    "example_id": key, "mode": "matched_budget",
                    "requested_budget": budget, "policy": "Random-Budget-Control", "seed": seed,
                    "routed": key in selected, "prediction": prediction,
                    "gold_label": gold_label(gold_rows[key]),
                })
        random_row = {
            "mode": "matched_budget", "requested_budget": budget,
            "requested_count": count, "policy": "Random-Budget-Control_mean",
        }
        for name in ("routed", "route_rate", "accuracy", "macro_f1", "helpful", "harmful", "net", "avg_model_forward_passes"):
            random_row[name] = statistics.mean(row[name] for row in random_budget_metrics)
            random_row[f"{name}_std"] = statistics.pstdev(row[name] for row in random_budget_metrics)
        matched_rows.append(random_row)
        quality_rows.append(random_row)
        confidence_scores = {key: features[key].baseline_entropy for key in ids}
        confidence_selected = exact_count(ids, confidence_scores, count)
        confidence_metrics, confidence_predictions = evaluate(
            ids, gold_rows, eval_baseline, eval_candidate, confidence_selected
        )
        confidence_row = {
            "mode": "matched_budget", "requested_budget": budget,
            "requested_count": count, "policy": "Confidence-Control",
            **{key: value for key, value in confidence_metrics.items() if key != "per_class"},
        }
        matched_rows.append(confidence_row)
        quality_rows.append(confidence_row)
        for key, prediction in zip(ids, confidence_predictions):
            prediction_rows.append({
                "example_id": key, "mode": "matched_budget", "requested_budget": budget,
                "policy": "Confidence-Control", "routed": key in confidence_selected,
                "prediction": prediction, "gold_label": gold_label(gold_rows[key]),
            })

    oracle = oracle_analysis(
        ids, gold_rows, eval_baseline, eval_candidate,
        routes["D0_HardLabel"], routes["D1_NLF"]
    )
    better_fixed_accuracy = max(natural_metrics["D0_HardLabel"]["accuracy"], natural_metrics["D1_NLF"]["accuracy"])
    oracle["potential_accuracy_over_better_fixed"] = oracle["accuracy_oracle"]["accuracy"] - better_fixed_accuracy

    expected = [gold_label(gold_rows[key]) for key in ids]
    primary_pairs = {
        "M2_vs_HardLabel": ("D0_HardLabel", "M2_Reliability"),
        "M2_vs_NLF": ("D1_NLF", "M2_Reliability"),
        "M3a_vs_HardLabel": ("D0_HardLabel", "M3a_DisagreementGate"),
        "M3a_vs_NLF": ("D1_NLF", "M3a_DisagreementGate"),
    }
    comparisons, raw_p = {}, {}
    cluster_groups = [component_by_id[key] for key in ids] if component_by_id else None
    for name, (left_name, right_name) in primary_pairs.items():
        left, right = natural_predictions[left_name], natural_predictions[right_name]
        report = {
            "orientation": "right_minus_left",
            "mcnemar_exact": mcnemar_exact(expected, left, right),
            "paired_bootstrap": paired_bootstrap(expected, left, right, args.bootstrap_samples, args.seed),
        }
        if cluster_groups:
            report["cluster_aware_bootstrap"] = cluster_bootstrap(
                expected, left, right, cluster_groups, args.bootstrap_samples, args.seed
            )
        comparisons[name] = report
        raw_p[name] = report["mcnemar_exact"]["exact_two_sided_p"]
    for name, adjusted in holm_adjust(raw_p).items():
        comparisons[name]["holm_adjusted_mcnemar_p"] = adjusted
    primary_random_seed = args.seed if args.seed in random_predictions_by_seed else args.random_seeds[0]
    secondary_pairs = {
        "M1_vs_HardLabel": (natural_predictions["D0_HardLabel"], natural_predictions["M1_MaxUtility"]),
        "M1_vs_NLF": (natural_predictions["D1_NLF"], natural_predictions["M1_MaxUtility"]),
        "M2_vs_M1": (natural_predictions["M1_MaxUtility"], natural_predictions["M2_Reliability"]),
        "M3a_vs_M2": (natural_predictions["M2_Reliability"], natural_predictions["M3a_DisagreementGate"]),
        f"M1_vs_Random{primary_random_seed}": (random_predictions_by_seed[primary_random_seed], natural_predictions["M1_MaxUtility"]),
        f"M2_vs_Random{primary_random_seed}": (random_predictions_by_seed[primary_random_seed], natural_predictions["M2_Reliability"]),
        f"M3a_vs_Random{primary_random_seed}": (random_predictions_by_seed[primary_random_seed], natural_predictions["M3a_DisagreementGate"]),
    }
    secondary = {}
    for name, (left, right) in secondary_pairs.items():
        secondary[name] = {
            "orientation": "right_minus_left",
            "mcnemar_exact": mcnemar_exact(expected, left, right),
            "paired_bootstrap": paired_bootstrap(
                expected, left, right, args.bootstrap_samples, args.seed
            ),
        }
        if cluster_groups:
            secondary[name]["cluster_aware_bootstrap"] = cluster_bootstrap(
                expected, left, right, cluster_groups, args.bootstrap_samples, args.seed
            )
    statistical_report = {
        "primary_holm_family": comparisons,
        "secondary_unadjusted": secondary,
        "oracle_headroom": {
            "potential_accuracy_over_better_fixed": oracle["potential_accuracy_over_better_fixed"],
            "accuracy_oracle": oracle["accuracy_oracle"]["accuracy"],
            "macro_f1_relaxed_upper_bound": oracle["macro_f1_diagnostic"]["relaxed_classwise_upper_bound"],
        },
    }

    reference_replay = {"status": "not_supplied"}
    if args.reference_summary:
        reference = json.loads(args.reference_summary.read_text(encoding="utf-8"))
        expected_h = reference["results"]["natural"]["R6-HardLabel"]
        expected_n = reference["results"]["natural"]["R6-NLF"]
        differences = {
            "hard_accuracy": natural_metrics["D0_HardLabel"]["accuracy"] - expected_h["accuracy"],
            "hard_macro_f1": natural_metrics["D0_HardLabel"]["macro_f1"] - expected_h["macro_f1"],
            "hard_routed": natural_metrics["D0_HardLabel"]["routed"] - expected_h["routed"],
            "nlf_accuracy": natural_metrics["D1_NLF"]["accuracy"] - expected_n["accuracy"],
            "nlf_macro_f1": natural_metrics["D1_NLF"]["macro_f1"] - expected_n["macro_f1"],
            "nlf_routed": natural_metrics["D1_NLF"]["routed"] - expected_n["routed"],
        }
        reference_replay = {"status": "ok" if all(abs(value) < 1e-12 for value in differences.values()) else "mismatch",
                            "differences": differences}
        if reference_replay["status"] != "ok":
            raise ValueError(f"Frozen reference replay mismatch: {differences}")

    audit = {
        "backbone_id": args.backbone_id, "dataset_protocol": args.dataset_protocol,
        "canonical_claim_tokenizer": CANONICAL_TOKENIZER,
        "tokenizer_matches_repository": CANONICAL_TOKENIZER == TOKEN_RE.pattern == r"[\w'-]+",
        "input_id_match": set(gold_rows) == set(baseline) == set(candidate),
        "memory_ids_unique": len(memory_ids) == len(set(memory_ids)),
        "memory_train_only": all(record.source_split == "train" for record in records),
        "external_target_memory_id_overlap": external_overlap,
        "memory_outcome_replay": memory_check,
        "b1_a3_manifest_model_values": sorted(model_values),
        "memory_candidate_manifest_model": memory_model,
        "backbone_memory_model_match": not model_values or not memory_model or memory_model in model_values,
        "reference_replay": reference_replay,
        "fold_exclusion": "query outer fold excluded from every source lookup" if assignments else "external source-only memory",
        "m1_union_exact_route_match": routes["D2_Union"] == routes["M1_MaxUtility"],
        "m2_beta0_m1_exact_decision_match": not beta0_mismatches,
        "m2_applied_beta_counts": dict(applied_beta_counts),
        "m2_zero_applied_beta_queries": len(zero_beta_ids),
        "m2_zero_applied_beta_m1_route_mismatches": len(zero_beta_actual_mismatches),
        "m2_actual_m1_route_mismatches": len(actual_m2_m1_route_mismatches),
        "single_a3_action_per_routed_claim": True,
        "gold_used_for_online_routing": False,
        "candidate_prediction_used_for_online_routing": False,
        "frozen_inputs_unchanged": all(
            sha256_file(path) == input_hashes_before[name]
            for name, path in input_paths.items()
        ),
        "input_hashes": {
            name: {"path": str(path), "sha256": input_hashes_before[name]}
            for name, path in input_paths.items()
        },
    }
    configs = {
        "backbone_id": args.backbone_id, "dataset_protocol": args.dataset_protocol,
        "protocol": args.protocol, "action": args.action, "top_k": args.top_k,
        "min_support": args.min_support, "cost_weight": args.cost_weight,
        "utility_threshold": args.utility_threshold, "group_threshold": args.group_threshold,
        "seed": args.seed, "beta_grid": args.beta_grid,
        "label_projection": projection,
        "source_selected_deployable_beta": deployable_beta,
        "grouped_internal_m2_evaluation": (
            "cross_fitted_beta_selected_on_complementary_outer_folds"
            if beta_by_fold is not None else None
        ),
        "source_selected_majority_action": deployable_majority,
        "source_selected_confidence_threshold": deployable_confidence,
        "majority_action_by_outer_fold": majority_by_fold,
        "confidence_threshold_by_outer_fold": confidence_by_fold,
        "cross_fitted_evaluation_beta_by_outer_fold": beta_by_fold,
        "beta_by_outer_fold": beta_by_fold,
        "beta_selection": selection,
        "M3a_rule": "retain agreement; on disagreement route only if the routing side's U-beta*SE remains positive",
        "M3b": "not_implemented_optional_comparator",
        "command_line": [sys.executable, *sys.argv],
        "score_comparability_audit": {
            name: {
                "eligible": sum(getattr(features[key], name).eligible for key in ids),
                "mean_support": statistics.mean(getattr(features[key], name).support for key in ids),
                "mean_effective_sample_size": statistics.mean(
                    getattr(features[key], name).effective_sample_size for key in ids
                ),
                "mean_standard_error": safe_mean(
                    getattr(features[key], name).standard_error
                    for key in ids if math.isfinite(getattr(features[key], name).standard_error)
                ),
                "mean_finite_utility": safe_mean(
                    getattr(features[key], name).utility
                    for key in ids if math.isfinite(getattr(features[key], name).utility)
                ),
            } for name in ("hard", "nlf")
        },
    }

    per_class_rows = []
    for policy, metrics in natural_metrics.items():
        for label, values in metrics["per_class"].items():
            per_class_rows.append({"mode": "natural", "policy": policy, "label": label, **values})
    for row in random_metrics:
        for label, values in row["per_class"].items():
            per_class_rows.append({
                "mode": "natural", "policy": "D4_Random", "seed": row["seed"],
                "label": label, **values,
            })
    for label in sorted({label for row in random_metrics for label in row["per_class"]}):
        f1_values = [row["per_class"][label]["f1"] for row in random_metrics]
        per_class_rows.append({
            "mode": "natural", "policy": "D4_Random_mean", "label": label,
            "f1": statistics.mean(f1_values), "f1_std": statistics.pstdev(f1_values),
            "support": random_metrics[0]["per_class"][label]["support"],
        })
    disagreement_rows = []
    for key in ids:
        h, n = decisions[key]["D0_HardLabel"].route, decisions[key]["D1_NLF"].route
        if h != n:
            gold = gold_label(gold_rows[key])
            before, after = eval_baseline[key]["prediction"], eval_candidate[key]["prediction"]
            disagreement_rows.append({
                "example_id": key, "hard_route": h, "nlf_route": n,
                "b1_prediction": before, "a3_prediction": after, "gold_label": gold,
                "b1_correct": before == gold, "a3_correct": after == gold,
                "hard_utility": features[key].hard.utility, "nlf_utility": features[key].nlf.utility,
                "utility_difference_nlf_minus_hard": features[key].utility_difference_nlf_minus_hard,
            })
    runtime_rows = [
        {"component": "hardlabel_memory_lookup", "mean_ms": statistics.mean(features[key].hard_lookup_ms for key in ids),
         "p95_ms": percentile([features[key].hard_lookup_ms for key in ids], 0.95), "n": len(ids),
         "scope": "CPU retrieval only; excludes B1/A3"},
        {"component": "nlf_memory_lookup", "mean_ms": statistics.mean(features[key].nlf_lookup_ms for key in ids),
         "p95_ms": percentile([features[key].nlf_lookup_ms for key in ids], 0.95), "n": len(ids),
         "scope": "CPU retrieval only; excludes B1/A3"},
        {"component": "dual_neighborhood_feature_extraction", "mean_ms": statistics.mean(latencies),
         "p95_ms": percentile(latencies, 0.95), "n": len(latencies),
         "scope": "CPU retrieval plus feature construction; excludes B1/A3"},
    ]

    write_jsonl(args.output_dir / "per_query_features.jsonl", feature_rows)
    write_jsonl(args.output_dir / "per_query_predictions.jsonl", prediction_rows)
    atomic_write_json(args.output_dir / "oracle_headroom.json", oracle)
    atomic_write_json(args.output_dir / "method_configs.json", configs)
    atomic_write_json(args.output_dir / "statistical_comparisons.json", statistical_report)
    write_csv(args.output_dir / "natural_results.csv", result_rows)
    write_csv(args.output_dir / "matched_budget_results.csv", matched_rows)
    write_csv(args.output_dir / "quality_budget_curves.csv", quality_rows)
    write_csv(args.output_dir / "per_class_results.csv", per_class_rows)
    write_csv(args.output_dir / "disagreement_analysis.csv", disagreement_rows)
    write_csv(args.output_dir / "runtime_comparison.csv", runtime_rows)
    audit_lines = ["# Meta-routing audit", "", "```json", json.dumps(audit, indent=2), "```", ""]
    (args.output_dir / "audit_report.md").write_text("\n".join(audit_lines), encoding="utf-8")
    report_lines = [
        "# ERRORFlow V3 experimental meta-routing benchmark", "",
        "This is an exploratory experiment. It does not replace R6-NLF or HardLabel.", "",
        f"- Backbone: `{args.backbone_id}`", f"- Dataset/protocol: `{args.dataset_protocol}`",
        f"- Source-selected beta: `{deployable_beta}`", "",
        "## A. Oracle headroom", "", f"- Disagreements: {oracle['disagreement']}",
        f"- Correctness-affecting disagreements: {oracle['correctness_affecting_disagreements']}",
        f"- Accuracy oracle: {oracle['accuracy_oracle']['accuracy']:.6f}",
        f"- Potential over better fixed policy: {oracle['potential_accuracy_over_better_fixed']:.6f}", "",
        "## B. M1 vs M2 vs M3", "",
    ]
    for row in result_rows:
        report_lines.append(
            f"- {row['policy']}: routes={row['routed']:.3f}, acc={row['accuracy']:.6f}, "
            f"macro-F1={row['macro_f1']:.6f}, net={row['net']:.3f}"
        )
    report_lines.extend((
        "", "Matched-budget curves and paired tests are stored in "
        "`matched_budget_results.csv` and `statistical_comparisons.json`.", "",
        "## C. Backbone robustness", "",
        "This run covers one backbone. Compare the identically generated Qwen and Mistral run directories before drawing a robustness conclusion.", "",
        "## D. Cross-domain robustness", "",
        "This run covers one evidence protocol. External runs reuse the source-selected beta and controls; target labels are read only after routes are frozen.", "",
        "## E. Mechanistic interpretation", "",
        f"Hard/ NLF score-comparability audit: `{json.dumps(configs['score_comparability_audit'], sort_keys=True)}`", "",
        "Per-query utilities, uncertainty, effective sample size, overlap, similarity margins, and cross-label mass are in `per_query_features.jsonl`.", "",
        "## F. Recommendation", "",
    ))
    fixed_best_acc = max(
        natural_metrics["D0_HardLabel"]["accuracy"], natural_metrics["D1_NLF"]["accuracy"]
    )
    fixed_best_f1 = max(
        natural_metrics["D0_HardLabel"]["macro_f1"], natural_metrics["D1_NLF"]["macro_f1"]
    )
    for policy in ("M1_MaxUtility", "M2_Reliability", "M3a_DisagreementGate"):
        row = natural_metrics[policy]
        if row["accuracy"] >= fixed_best_acc and row["macro_f1"] >= fixed_best_f1:
            recommendation = "promising for further evaluation"
        elif row["accuracy"] < min(
            natural_metrics["D0_HardLabel"]["accuracy"], natural_metrics["D1_NLF"]["accuracy"]
        ) and row["macro_f1"] < min(
            natural_metrics["D0_HardLabel"]["macro_f1"], natural_metrics["D1_NLF"]["macro_f1"]
        ):
            recommendation = "inferior to fixed routing on this run"
        else:
            recommendation = "inconclusive"
        report_lines.append(f"- {policy}: **{recommendation}** (descriptive; consult paired tests).")
    report_lines.extend(("", "No method is automatically adopted. External target labels are used only after routes are frozen.", ""))
    (args.output_dir / "experiment_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir), "audit": audit,
        "deployable_beta": deployable_beta, "oracle_headroom": oracle,
        "natural_results": natural_metrics, "comparisons": statistical_report,
    }, indent=2))
    print(f"META_ROUTING_DONE rows={len(ids)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
