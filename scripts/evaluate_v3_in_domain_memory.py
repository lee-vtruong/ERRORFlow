"""Fold-safe in-domain outcome-memory evaluation from cached Base/Alt runs.

This is an isolated diagnostic experiment.  Every query excludes its complete
near-duplicate group fold from memory before route scoring.  Gold labels are
used to construct outcomes in the other folds and to evaluate materialized
predictions, never as query-time routing features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_memory import ResidualMemoryStore, build_action_memories
from errorflow.v3_residuals import build_residual_records
from errorflow.v3_routing import best_memory_action, deterministic_random_score
from errorflow.v3_statistics import (
    grouped_paired_bootstrap,
    mcnemar_exact,
    paired_bootstrap,
    percentile,
)

try:
    from scripts.evaluate_v3_grouping_sensitivity import grouped_folds, near_duplicate_edges
except ModuleNotFoundError:
    from evaluate_v3_grouping_sensitivity import grouped_folds, near_duplicate_edges


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


def finite_top_count(ids, scores, count):
    eligible = [key for key in ids if math.isfinite(scores.get(key, float("-inf")))]
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} eligible scores for matched count {count}")
    return set(sorted(eligible, key=lambda key: (-scores[key], key))[:count])


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


def evaluate(ids, residuals, candidates, selected):
    gold = [residuals[key]["gold_label"] for key in ids]
    before = [residuals[key]["prediction"] for key in ids]
    after = [candidates[key]["prediction"] if key in selected else residuals[key]["prediction"] for key in ids]
    helpful = sum(b != g and a == g for g, b, a in zip(gold, before, after))
    harmful = sum(b == g and a != g for g, b, a in zip(gold, before, after))
    forward_passes = [
        float(residuals[key].get("model_forward_passes", 4.0))
        + (
            float(candidates[key].get("metadata", {}).get("model_forward_passes", 4.0))
            if key in selected else 0.0
        )
        for key in ids
    ]
    return {
        "n": len(ids),
        "routed": len(selected),
        "route_rate": len(selected) / len(ids),
        "accuracy": accuracy(after, gold),
        "macro_f1": macro_f1(after, gold),
        "per_class": per_class(gold, after),
        "helpful": helpful,
        "harmful": harmful,
        "net": helpful - harmful,
        "avg_model_forward_passes": sum(forward_passes) / len(forward_passes),
        "normalized_prompt_requests": 1.0 + len(selected) / len(ids),
    }, after


def compare(gold, left, right, samples, seed, groups=None):
    bootstrap = paired_bootstrap(gold, left, right, samples, seed)
    bootstrap["delta_accuracy"] = accuracy(right, gold) - accuracy(left, gold)
    bootstrap["delta_macro_f1"] = macro_f1(right, gold) - macro_f1(left, gold)
    result = {
        "orientation": "right_minus_left",
        "mcnemar_exact": mcnemar_exact(gold, left, right),
        "paired_bootstrap": bootstrap,
    }
    if groups is not None:
        result["grouped_paired_bootstrap"] = grouped_paired_bootstrap(
            gold, left, right, groups, samples, seed
        )
    return result


def uncertainty(row):
    probabilities = row.get("label_probabilities", {})
    if probabilities:
        return 1.0 - max(float(value) for value in probabilities.values())
    if row.get("prediction_margin") is not None:
        return -float(row["prediction_margin"])
    raise ValueError(f"Missing uncertainty features for {row['example_id']}")


def route_scores(ids, residuals, store, action, top_k, min_support, cost_weight, mode):
    scores, natural, diagnostics, timings = {}, set(), {}, []
    for index, key in enumerate(ids, 1):
        row = residuals[key]
        query = {
            "example_id": key,
            "claim": row["claim"],
            "retrieval_features": {
                "baseline_prediction": row["prediction"],
                "label_scores": row.get("label_scores", {}),
                "evidence_tokens": row.get("evidence_tokens", 0),
            },
        }
        started = time.perf_counter()
        neighbors = store.retrieve(
            query,
            top_k=top_k,
            action=action,
            exclude_claim_ids={key},
            exclude_folds={row["outer_fold"]},
            compatibility_mode=mode,
        )
        timings.append((time.perf_counter() - started) * 1000.0)
        best, estimates = best_memory_action([action], neighbors, cost_weight, min_support, True)
        score = best.utility if best else float("-inf")
        scores[key] = score
        if best and score > 0.0:
            natural.add(key)
        diagnostics[key] = {
            "outer_fold": row["outer_fold"],
            "utility": score if math.isfinite(score) else None,
            "support": best.support if best else 0,
            "neighbors": [
                {
                    "source_claim_id": record.source_claim_id,
                    "source_fold": record.provenance.get("outer_fold"),
                    "similarity": similarity,
                    "recovered": record.recovery_count,
                    "regressed": record.regression_count,
                }
                for record, similarity in neighbors
            ],
            "estimates": [estimate.to_dict() for estimate in estimates],
        }
        if index % 250 == 0 or index == len(ids):
            print(f"ROUTING mode={mode} progress={index}/{len(ids)} routes={len(natural)}", flush=True)
    return scores, natural, diagnostics, {
        "mean_lookup_ms": sum(timings) / len(timings),
        "p95_lookup_ms": percentile(timings, 0.95),
        "measurement": "single-process CPU retrieval only; excludes model inference",
    }


def fold_report(ids, assignments, residuals, candidates, selected, folds):
    report = []
    for fold in range(folds):
        members = [key for key in ids if assignments[key] == fold]
        metrics, _ = evaluate(members, residuals, candidates, selected & set(members))
        report.append({"outer_fold": fold, **metrics})
    return report


def aggregate_random(seed_results):
    fields = ("accuracy", "macro_f1", "helpful", "harmful", "net")
    result = {"seeds": len(seed_results)}
    for field in fields:
        values = [float(row[field]) for row in seed_results]
        mean = sum(values) / len(values)
        result[field] = {
            "mean": mean,
            "min": min(values),
            "max": max(values),
            "std": math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--action", default="conflicting_rescue_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--group-threshold", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=list(range(2026, 2046)))
    args = parser.parse_args(argv)
    if args.folds < 2:
        raise ValueError("--folds must be at least two")
    if not 0.0 <= args.group_threshold <= 1.0:
        raise ValueError("--group-threshold must be within [0, 1]")
    if args.seed not in args.random_seeds:
        raise ValueError("--random-seeds must include --seed for the declared primary random control")

    gold_rows = list(read_jsonl(args.gold))
    baseline_rows = list(read_jsonl(args.baseline))
    candidate_rows = list(read_jsonl(args.candidate))
    gold = indexed(gold_rows, "gold")
    baseline = indexed(baseline_rows, "baseline")
    candidates = indexed(candidate_rows, "candidate")
    if set(gold) != set(baseline) or set(gold) != set(candidates):
        raise ValueError("Gold, baseline, and candidate IDs must match exactly")
    datasets = {str(row.get("dataset")) for row in gold_rows}
    if len(datasets) != 1:
        raise ValueError(f"Expected exactly one dataset, found {sorted(datasets)}")
    dataset = args.dataset or next(iter(datasets))

    residual_rows = build_residual_records(gold_rows, baseline_rows)
    edges = near_duplicate_edges(residual_rows, args.group_threshold)
    assignments, components = grouped_folds(
        residual_rows, edges, args.group_threshold, args.folds, args.seed,
    )
    for row in residual_rows:
        row["outer_fold"] = assignments[str(row["example_id"])]
    records = build_action_memories(
        residual_rows,
        [(args.action, candidate_rows)],
        dataset=dataset,
        source_protocol="grouped_in_domain",
    )
    store = ResidualMemoryStore(records)
    residuals = indexed(residual_rows, "residuals")
    ids = sorted(residuals)

    nlf_scores, nlf_natural, nlf_diag, nlf_timing = route_scores(
        ids, residuals, store, args.action, args.top_k, args.min_support,
        args.cost_weight, "none",
    )
    hard_scores, hard_natural, hard_diag, hard_timing = route_scores(
        ids, residuals, store, args.action, args.top_k, args.min_support,
        args.cost_weight, "same_label",
    )
    count = len(nlf_natural)
    confidence_scores = {key: uncertainty(residuals[key]) for key in ids}
    selections = {
        "Base": set(),
        "Alt-All": set(ids),
        "Memory-Global-natural": nlf_natural,
        "Memory-SameLabel-natural": hard_natural,
        "Memory-SameLabel-matched": finite_top_count(ids, hard_scores, count),
        "Confidence-matched": finite_top_count(ids, confidence_scores, count),
    }
    random_metrics, random_selections = [], {}
    for random_seed in args.random_seeds:
        scores = {key: deterministic_random_score(key, random_seed) for key in ids}
        selected = finite_top_count(ids, scores, count)
        random_selections[random_seed] = selected
        metrics, _ = evaluate(ids, residuals, candidates, selected)
        random_metrics.append(metrics)
    selections[f"Random-matched-seed-{args.seed}"] = random_selections[args.seed]

    results, predictions, prediction_rows, route_rows = {}, {}, [], []
    for policy, selected in selections.items():
        metrics, output = evaluate(ids, residuals, candidates, selected)
        results[policy] = metrics
        predictions[policy] = output
        route_rows.append({"policy": policy, "selected_ids": sorted(selected)})
        for key, prediction in zip(ids, output):
            prediction_rows.append({
                "example_id": key,
                "outer_fold": assignments[key],
                "policy": policy,
                "routed": key in selected,
                "gold_label": residuals[key]["gold_label"],
                "baseline_prediction": residuals[key]["prediction"],
                "candidate_prediction": candidates[key]["prediction"],
                "prediction": prediction,
            })

    expected = [residuals[key]["gold_label"] for key in ids]
    component_by_id = {
        key: min(component) for component in components for key in component
    }
    bootstrap_groups = [component_by_id[key] for key in ids]
    primary = "Memory-Global-natural"
    comparisons = {
        "Memory-Global_vs_Base": compare(
            expected, predictions["Base"], predictions[primary], args.samples, args.seed,
            bootstrap_groups,
        ),
        "Memory-Global_vs_Alt-All": compare(
            expected, predictions["Alt-All"], predictions[primary], args.samples, args.seed,
            bootstrap_groups,
        ),
        "Memory-Global_vs_Confidence": compare(
            expected, predictions["Confidence-matched"], predictions[primary], args.samples, args.seed,
            bootstrap_groups,
        ),
        "Memory-Global_vs_SameLabel": compare(
            expected, predictions["Memory-SameLabel-matched"], predictions[primary], args.samples, args.seed,
            bootstrap_groups,
        ),
        "Memory-Global_vs_Random2026": compare(
            expected, predictions[f"Random-matched-seed-{args.seed}"], predictions[primary],
            args.samples, args.seed, bootstrap_groups,
        ),
    }
    isolation = []
    for fold in range(args.folds):
        validation = {key for key, value in assignments.items() if value == fold}
        usable = {
            record.source_claim_id for record in records
            if record.provenance.get("outer_fold") != fold
        }
        isolation.append({
            "outer_fold": fold,
            "validation_examples": len(validation),
            "usable_memory_records": len(usable),
            "usable_validation_id_overlap": len(validation & usable),
        })
    if any(row["usable_validation_id_overlap"] for row in isolation):
        raise AssertionError("Fold leakage detected")

    base_accuracy = results["Base"]["accuracy"]
    memory_accuracy = results[primary]["accuracy"]
    memory_vs_base_ci = comparisons["Memory-Global_vs_Base"]["grouped_paired_bootstrap"]["accuracy_delta_ci95"]
    strict_support = (
        memory_accuracy > base_accuracy
        and memory_vs_base_ci[0] > 0.0
        and results[primary]["avg_model_forward_passes"] < results["Alt-All"]["avg_model_forward_passes"]
    )
    summary = {
        "evaluation": "grouped_in_domain_outcome_memory_crossfit",
        "dataset": dataset,
        "n": len(ids),
        "configuration": {
            "folds": args.folds,
            "grouping": "normalized_claim_token_jaccard_connected_components",
            "group_threshold": args.group_threshold,
            "top_k": args.top_k,
            "min_support": args.min_support,
            "cost_weight": args.cost_weight,
            "utility_threshold": 0.0,
            "seed": args.seed,
            "action": args.action,
        },
        "grouping": {
            "components": len(components),
            "multi_example_components": sum(len(component) > 1 for component in components),
            "largest_component": max(map(len, components)),
            "fold_sizes": dict(Counter(assignments.values())),
        },
        "fold_isolation": isolation,
        "results": results,
        "comparisons": comparisons,
        "random_matched_budget": aggregate_random(random_metrics),
        "fold_results": fold_report(
            ids, assignments, residuals, candidates, nlf_natural, args.folds,
        ),
        "retrieval_overhead": {
            "Memory-Global": nlf_timing,
            "Memory-SameLabel": hard_timing,
        },
        "hypothesis_test": {
            "criterion": "Memory-Global improves Base with positive group-bootstrap accuracy CI and uses fewer forwards than Alt-All",
            "supported_under_this_protocol": strict_support,
            "descriptive_accuracy_gain": memory_accuracy - base_accuracy,
            "accuracy_delta_ci95": memory_vs_base_ci,
            "resampling_unit": "near_duplicate_group",
            "memory_avg_forward_passes": results[primary]["avg_model_forward_passes"],
            "alt_all_avg_forward_passes": results["Alt-All"]["avg_model_forward_passes"],
        },
        "input_artifacts": {
            "gold": {"path": str(args.gold), "sha256": sha256_file(args.gold)},
            "baseline": {"path": str(args.baseline), "sha256": sha256_file(args.baseline)},
            "candidate": {"path": str(args.candidate), "sha256": sha256_file(args.candidate)},
        },
        "interpretation_constraints": [
            "This is grouped out-of-fold source-domain evaluation, not an untouched held-out estimate.",
            "Each query excludes its complete near-duplicate group fold from memory.",
            "Gold labels create outcomes only in the other folds and are not query-time features.",
            "SciFact train-to-dev evaluation should be reported separately as the independent in-domain transfer test.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "baseline_scores.jsonl", residual_rows)
    store.save(args.output_dir / "memory.jsonl")
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(args.output_dir / "routes.jsonl", route_rows)
    write_jsonl(args.output_dir / "neighbor_diagnostics.jsonl", (
        {
            "example_id": key,
            "Memory-Global": nlf_diag[key],
            "Memory-SameLabel": hard_diag[key],
        }
        for key in ids
    ))
    atomic_write_json(args.output_dir / "summary.json", summary)
    atomic_write_json(args.output_dir / "manifest.json", {
        "schema_version": 1,
        "status": "complete",
        "evaluation": summary["evaluation"],
        "configuration": summary["configuration"],
        "input_artifacts": summary["input_artifacts"],
        "outputs": {
            "summary": str(args.output_dir / "summary.json"),
            "memory": str(args.output_dir / "memory.jsonl"),
            "predictions": str(args.output_dir / "predictions.jsonl"),
            "routes": str(args.output_dir / "routes.jsonl"),
            "neighbor_diagnostics": str(args.output_dir / "neighbor_diagnostics.jsonl"),
        },
    })
    print(json.dumps({
        "dataset": dataset,
        "results": results,
        "comparisons": comparisons,
        "random_matched_budget": summary["random_matched_budget"],
        "hypothesis_test": summary["hypothesis_test"],
    }, indent=2))
    print(f"IN_DOMAIN_MEMORY_DONE rows={len(ids)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
