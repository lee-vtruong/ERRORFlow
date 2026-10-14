"""Evaluate frozen R6-NLF against R6-HardLabel and matched controls.

The script supports grouped AVeriTeC development and frozen source-only
external evaluation.  Route construction never reads target gold labels.
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
from errorflow.v3_memory import ResidualMemoryStore, record_from_dict
from errorflow.v3_routing import best_memory_action
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap, percentile

try:
    from scripts.evaluate_v3_matched_budget import prepare_grouped_artifacts
except ModuleNotFoundError:
    from evaluate_v3_matched_budget import prepare_grouped_artifacts


POLICY_HARD = "R6-HardLabel"
POLICY_NLF = "R6-NLF"
POLICY_CONFIDENCE = "Confidence-Only"


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


def label_scores(row):
    return row.get("label_scores") or row.get("metadata", {}).get("label_scores", {})


def evidence_tokens(row):
    return int(row.get("evidence_tokens", row.get("metadata", {}).get("evidence_tokens", 0)))


def gold_label(row):
    value = row.get("label", row.get("gold_label"))
    if value is None:
        raise ValueError(f"Missing gold label for {row.get('example_id')}")
    return value


def uncertainty(row):
    if row.get("confidence") is not None:
        return 1.0 - float(row["confidence"])
    scores = label_scores(row)
    if not scores:
        raise ValueError(f"Missing confidence and label scores for {row['example_id']}")
    maximum = max(float(value) for value in scores.values())
    values = [math.exp(float(value) - maximum) for value in scores.values()]
    return 1.0 - max(values) / sum(values)


def query_from(key, row):
    return {
        "example_id": key,
        "claim": row["claim"],
        "retrieval_features": {
            "baseline_prediction": row["prediction"],
            "label_scores": label_scores(row),
            "evidence_tokens": evidence_tokens(row),
        },
    }


def select_count(ids, scores, count):
    finite = [key for key in ids if math.isfinite(scores.get(key, float("-inf")))]
    if len(finite) < count:
        raise ValueError(f"Only {len(finite)} finite route scores for requested count {count}")
    return set(sorted(finite, key=lambda key: (-scores[key], key))[:count])


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


def evaluate(ids, gold_rows, baseline, candidate, selected, projection):
    expected = [gold_label(gold_rows[key]) for key in ids]
    before = [projection.get(baseline[key]["prediction"], baseline[key]["prediction"]) for key in ids]
    raw_after = [candidate[key]["prediction"] if key in selected else baseline[key]["prediction"] for key in ids]
    after = [projection.get(value, value) for value in raw_after]
    helpful = sum(b != g and a == g for g, b, a in zip(expected, before, after))
    harmful = sum(b == g and a != g for g, b, a in zip(expected, before, after))
    latencies = [
        float(baseline[key].get("latency_ms", 0.0))
        + (float(candidate[key].get("latency_ms", 0.0)) if key in selected else 0.0)
        for key in ids
    ]
    return {
        "n": len(ids), "routed": len(selected), "route_rate": len(selected) / len(ids),
        "accuracy": accuracy(after, expected), "macro_f1": macro_f1(after, expected),
        "per_class": per_class(expected, after), "helpful": helpful,
        "harmful": harmful, "net": helpful - harmful,
        "avg_model_forward_passes": 4.0 + 4.0 * len(selected) / len(ids),
        "mean_cached_component_latency_ms": sum(latencies) / len(latencies),
    }, after


def comparison(gold, left, right, samples, seed):
    bootstrap = paired_bootstrap(gold, left, right, samples, seed)
    bootstrap["delta_accuracy"] = accuracy(right, gold) - accuracy(left, gold)
    bootstrap["delta_macro_f1"] = macro_f1(right, gold) - macro_f1(left, gold)
    return {
        "orientation": "right_minus_left",
        "mcnemar_exact": mcnemar_exact(gold, left, right),
        "paired_bootstrap": bootstrap,
    }


def route_scores(ids, baseline, store, action, top_k, min_support, cost_weight,
                 threshold, compatibility_mode, grouped):
    scores, natural, diagnostics, elapsed = {}, set(), {}, []
    for index, key in enumerate(ids, 1):
        row = baseline[key]
        excluded_folds = {row["outer_fold"]} if grouped else None
        started = time.perf_counter()
        neighbors = store.retrieve(
            query_from(key, row), top_k=top_k, action=action,
            exclude_claim_ids={key}, exclude_folds=excluded_folds,
            compatibility_mode=compatibility_mode,
        )
        elapsed.append((time.perf_counter() - started) * 1000.0)
        best, _ = best_memory_action([action], neighbors, cost_weight, min_support, True)
        scores[key] = best.utility if best else float("-inf")
        if best and best.utility > threshold:
            natural.add(key)
        query_label = row["prediction"]
        neighbor_labels = [record.retrieval_features.get("baseline_prediction") for record, _ in neighbors]
        cross = sum(label != query_label for label in neighbor_labels)
        diagnostics[key] = {
            "query_prediction": query_label,
            "neighbor_labels": neighbor_labels,
            "neighbor_ids": [record.source_claim_id for record, _ in neighbors],
            "neighbor_similarities": [score for _, score in neighbors],
            "cross_label_neighbors": cross,
            "has_cross_label_neighbor": cross > 0,
            "utility": scores[key] if math.isfinite(scores[key]) else None,
            "support": best.support if best else 0,
        }
        if index % 250 == 0 or index == len(ids):
            print(
                f"ROUTING policy={compatibility_mode} progress={index}/{len(ids)} "
                f"natural_routes={len(natural)}",
                flush=True,
            )
    overhead = {
        "mean_lookup_ms": sum(elapsed) / len(elapsed),
        "p95_lookup_ms": percentile(elapsed, 0.95),
        "total_lookup_s": sum(elapsed) / 1000.0,
        "measurement": "fresh single-process CPU wall clock around memory retrieval only; excludes B1/A3",
    }
    return scores, natural, diagnostics, overhead


def exclusive_flip_summary(keys, gold_rows, baseline, candidate, projection):
    helpful = harmful = changed = 0
    for key in keys:
        gold = gold_label(gold_rows[key])
        before = projection.get(baseline[key]["prediction"], baseline[key]["prediction"])
        after = projection.get(candidate[key]["prediction"], candidate[key]["prediction"])
        helpful += before != gold and after == gold
        harmful += before == gold and after != gold
        changed += before != after
    return {
        "n": len(keys), "helpful": helpful, "harmful": harmful,
        "net": helpful - harmful, "changed_prediction": changed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", choices=["external", "grouped_internal"], default="external")
    parser.add_argument("--action", default="conflicting_rescue_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--group-threshold", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--project-conflicting-to-nei", action="store_true")
    args = parser.parse_args()

    gold_rows = indexed(read_jsonl(args.gold), "gold")
    baseline_rows = list(read_jsonl(args.baseline))
    memory_rows = list(read_jsonl(args.memory))
    grouping = None
    grouped = args.protocol == "grouped_internal"
    if grouped:
        baseline_rows, records, _, grouping = prepare_grouped_artifacts(
            baseline_rows, memory_rows, args.group_threshold, args.folds, args.seed,
        )
        store = ResidualMemoryStore(records)
    else:
        store = ResidualMemoryStore(record_from_dict(row) for row in memory_rows)
    baseline = indexed(baseline_rows, "baseline")
    candidate = indexed(read_jsonl(args.candidate), "candidate")
    if set(gold_rows) != set(baseline) or set(gold_rows) != set(candidate):
        raise ValueError("Gold, baseline, and candidate IDs must match exactly")
    ids = sorted(gold_rows)
    hard_scores, hard_natural, hard_diag, hard_overhead = route_scores(
        ids, baseline, store, args.action, args.top_k, args.min_support,
        args.cost_weight, args.utility_threshold, "same_label", grouped,
    )
    nlf_scores, nlf_natural, nlf_diag, nlf_overhead = route_scores(
        ids, baseline, store, args.action, args.top_k, args.min_support,
        args.cost_weight, args.utility_threshold, "none", grouped,
    )
    confidence_scores = {key: uncertainty(baseline[key]) for key in ids}
    hard_count, nlf_count = len(hard_natural), len(nlf_natural)
    selections = {
        "anchors": {"B1": set(), "A3-All": set(ids)},
        "natural": {POLICY_HARD: hard_natural, POLICY_NLF: nlf_natural},
        "matched_nlf_budget": {
            POLICY_HARD: select_count(ids, hard_scores, nlf_count),
            POLICY_NLF: select_count(ids, nlf_scores, nlf_count),
            POLICY_CONFIDENCE: select_count(ids, confidence_scores, nlf_count),
        },
        "matched_hard_budget": {
            POLICY_HARD: select_count(ids, hard_scores, hard_count),
            POLICY_NLF: select_count(ids, nlf_scores, hard_count),
            POLICY_CONFIDENCE: select_count(ids, confidence_scores, hard_count),
        },
    }
    projection = {"CONFLICTING": "NOT ENOUGH INFO"} if args.project_conflicting_to_nei else {}
    expected = [gold_label(gold_rows[key]) for key in ids]
    results, outputs, prediction_rows, route_rows = {}, {}, [], []
    for mode, policies in selections.items():
        results[mode], outputs[mode] = {}, {}
        for policy, selected in policies.items():
            metrics, predictions = evaluate(ids, gold_rows, baseline, candidate, selected, projection)
            results[mode][policy] = metrics
            outputs[mode][policy] = predictions
            route_rows.append({"mode": mode, "policy": policy, "selected_ids": sorted(selected)})
            for key, prediction in zip(ids, predictions):
                raw_prediction = candidate[key]["prediction"] if key in selected else baseline[key]["prediction"]
                prediction_rows.append({
                    "example_id": key, "mode": mode, "policy": policy,
                    "routed": key in selected, "prediction": prediction,
                    "pre_projection_prediction": raw_prediction,
                    "projection_applied": raw_prediction != prediction,
                    "gold_label": gold_label(gold_rows[key]),
                })
    comparisons = {
        "natural_nlf_vs_hard": comparison(
            expected, outputs["natural"][POLICY_HARD], outputs["natural"][POLICY_NLF], args.samples, args.seed,
        ),
        "natural_nlf_vs_b1": comparison(
            expected, outputs["anchors"]["B1"], outputs["natural"][POLICY_NLF], args.samples, args.seed,
        ),
        "matched_nlf_budget_nlf_vs_hard": comparison(
            expected, outputs["matched_nlf_budget"][POLICY_HARD],
            outputs["matched_nlf_budget"][POLICY_NLF], args.samples, args.seed,
        ),
        "matched_nlf_budget_nlf_vs_confidence": comparison(
            expected, outputs["matched_nlf_budget"][POLICY_CONFIDENCE],
            outputs["matched_nlf_budget"][POLICY_NLF], args.samples, args.seed,
        ),
        "matched_hard_budget_nlf_vs_hard": comparison(
            expected, outputs["matched_hard_budget"][POLICY_HARD],
            outputs["matched_hard_budget"][POLICY_NLF], args.samples, args.seed,
        ),
    }
    only_nlf, only_hard = nlf_natural - hard_natural, hard_natural - nlf_natural
    route_difference = {
        "hard_natural_routes": hard_count, "nlf_natural_routes": nlf_count,
        "intersection": len(hard_natural & nlf_natural),
        "only_nlf": exclusive_flip_summary(only_nlf, gold_rows, baseline, candidate, projection),
        "only_hard": exclusive_flip_summary(only_hard, gold_rows, baseline, candidate, projection),
        "jaccard": len(hard_natural & nlf_natural) / len(hard_natural | nlf_natural)
        if hard_natural | nlf_natural else 1.0,
    }
    nlf_cross_total = sum(row["cross_label_neighbors"] for row in nlf_diag.values())
    nlf_cross_queries = {key for key, row in nlf_diag.items() if row["has_cross_label_neighbor"]}
    pair_counts = Counter()
    for row in nlf_diag.values():
        for neighbor_label in row["neighbor_labels"]:
            if neighbor_label != row["query_prediction"]:
                pair_counts[f"{row['query_prediction']}->{neighbor_label}"] += 1
    neighbor_analysis = {
        "n_queries": len(ids), "top_k": args.top_k,
        "total_returned_neighbors": sum(len(row["neighbor_ids"]) for row in nlf_diag.values()),
        "cross_label_neighbors": nlf_cross_total,
        "cross_label_neighbor_rate": nlf_cross_total /
        max(1, sum(len(row["neighbor_ids"]) for row in nlf_diag.values())),
        "queries_with_cross_label_neighbor": len(nlf_cross_queries),
        "query_rate_with_cross_label_neighbor": len(nlf_cross_queries) / len(ids),
        "routed_nlf_queries_with_cross_label_neighbor": len(nlf_natural & nlf_cross_queries),
        "routed_nlf_cross_label_query_rate": len(nlf_natural & nlf_cross_queries) / max(1, len(nlf_natural)),
        "cross_label_pairs": dict(pair_counts),
    }
    diagnostic_rows = []
    for key in ids:
        diagnostic_rows.append({
            "example_id": key, "hard": hard_diag[key], "nlf": nlf_diag[key],
            "hard_natural_route": key in hard_natural,
            "nlf_natural_route": key in nlf_natural,
            "route_membership": "both" if key in hard_natural and key in nlf_natural else
            ("nlf_only" if key in nlf_natural else ("hard_only" if key in hard_natural else "neither")),
        })
    config_payload = {
        "policy_id": "errorflow_r6_nlf_v1", "top_k": args.top_k,
        "min_support": args.min_support, "cost_weight": args.cost_weight,
        "utility_threshold": args.utility_threshold,
        "similarity_weights": [0.35, 0.50, 0.15],
        "compatibility_mode": "none", "action": args.action,
        "protocol": args.protocol, "group_threshold": args.group_threshold if grouped else None,
        "folds": args.folds if grouped else None, "seed": args.seed,
    }
    config_id = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()[:32]
    summary = {
        "evaluation": "frozen_r6_nlf_vs_hardlabel",
        "config_id": config_id, "configuration": config_payload,
        "target_labels_used_for_routing": False, "grouping": grouping,
        "results": results, "comparisons": comparisons,
        "route_difference": route_difference,
        "nlf_neighbor_analysis": neighbor_analysis,
        "retrieval_overhead": {POLICY_HARD: hard_overhead, POLICY_NLF: nlf_overhead},
        "input_artifacts": {
            "gold": {"path": str(args.gold), "sha256": sha256_file(args.gold)},
            "baseline": {"path": str(args.baseline), "sha256": sha256_file(args.baseline)},
            "candidate": {"path": str(args.candidate), "sha256": sha256_file(args.candidate)},
            "memory": {"path": str(args.memory), "sha256": sha256_file(args.memory)},
        },
        "interpretation_constraints": [
            "R6-NLF was selected on AVeriTeC grouped development ablations.",
            "External target labels are read only after all routes are materialized.",
            "Matched-NLF-budget is the primary router comparison; matched-HardLabel-budget is sensitivity.",
            "Memory lookup timing is fresh CPU wall clock and is not synchronized GPU end-to-end latency.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "policy_manifest.json", {
        "schema_version": 1,
        "status": "complete",
        "config_id": config_id,
        "configuration": config_payload,
        "target_labels_used_for_routing": False,
        "input_artifacts": summary["input_artifacts"],
        "outputs": {
            "summary": str(args.output_dir / "summary.json"),
            "predictions": str(args.output_dir / "predictions.jsonl"),
            "routes": str(args.output_dir / "routes.jsonl"),
            "neighbor_diagnostics": str(args.output_dir / "neighbor_diagnostics.jsonl"),
        },
    })
    atomic_write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(args.output_dir / "routes.jsonl", route_rows)
    write_jsonl(args.output_dir / "neighbor_diagnostics.jsonl", diagnostic_rows)
    print(json.dumps({
        "config_id": config_id, "results": results,
        "route_difference": route_difference,
        "nlf_neighbor_analysis": neighbor_analysis,
        "retrieval_overhead": summary["retrieval_overhead"],
        "comparisons": comparisons,
    }, indent=2))
    print(f"NLF_EVALUATION_DONE rows={len(ids)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
