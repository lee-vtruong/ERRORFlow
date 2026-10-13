"""Evaluate frozen source-only V3 routers on a four-label external dataset."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_memory import ResidualMemoryStore
from errorflow.v3_routing import best_memory_action
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap


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


def label_scores_from(row):
    return row.get("label_scores") or row.get("metadata", {}).get("label_scores", {})


def evidence_tokens_from(row):
    return int(row.get("evidence_tokens", row.get("metadata", {}).get("evidence_tokens", 0)))


def uncertainty_from(row):
    if row.get("confidence") is not None:
        return 1.0 - float(row["confidence"])
    scores = label_scores_from(row)
    if not scores:
        raise ValueError(f"Missing confidence and label scores for {row['example_id']}")
    maximum = max(float(value) for value in scores.values())
    weights = [math.exp(float(value) - maximum) for value in scores.values()]
    return 1.0 - max(weights) / sum(weights)


def query_from(row):
    return {
        "claim": row["claim"],
        "retrieval_features": {
            "baseline_prediction": row["prediction"],
            "label_scores": label_scores_from(row),
            "evidence_tokens": evidence_tokens_from(row),
        },
    }


def select_count(ids, scores, count):
    return set(sorted(ids, key=lambda key: (-scores.get(key, float("-inf")), key))[:count])


def build_routes(ids, baseline, store, action, top_k, min_support, cost_weight, threshold):
    confidence, similarity, utility = {}, {}, {}
    positive = set()
    for key in ids:
        row = baseline[key]
        confidence[key] = uncertainty_from(row)
        neighbors = store.retrieve(query_from(row), top_k=top_k, action=action)
        similarity[key] = max((score for _, score in neighbors), default=float("-inf"))
        best, _ = best_memory_action([action], neighbors, cost_weight, min_support, True)
        utility[key] = best.utility if best else float("-inf")
        if best and best.utility > threshold:
            positive.add(key)
    count = len(positive)
    return {
        "B1": set(), "A3-All": set(ids),
        "R-Confidence": select_count(ids, confidence, count),
        "R-Similarity": select_count(ids, similarity, count),
        "R6": positive,
    }, {"confidence": confidence, "similarity": similarity, "utility": utility}


def per_class(gold, predicted):
    result = {}
    for label in sorted(set(gold) | set(predicted)):
        tp = sum(g == p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result[label] = {
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": sum(g == label for g in gold),
        }
    return result


def evaluate(ids, gold_rows, baseline, candidate, selected):
    gold = [gold_rows[key]["label"] for key in ids]
    before = [baseline[key]["prediction"] for key in ids]
    after = [candidate[key]["prediction"] if key in selected else baseline[key]["prediction"] for key in ids]
    helpful = sum(b != g and a == g for g, b, a in zip(gold, before, after))
    harmful = sum(b == g and a != g for g, b, a in zip(gold, before, after))
    latencies = [
        float(baseline[key].get("latency_ms", 0.0))
        + (float(candidate[key].get("latency_ms", 0.0)) if key in selected else 0.0)
        for key in ids
    ]
    return {
        "n": len(ids), "routed": len(selected), "route_rate": len(selected) / len(ids),
        "accuracy": accuracy(after, gold), "macro_f1": macro_f1(after, gold),
        "per_class": per_class(gold, after), "helpful": helpful,
        "harmful": harmful, "net": helpful - harmful,
        "avg_model_forward_passes": 4.0 + 4.0 * len(selected) / len(ids),
        "mean_cached_component_latency_ms": sum(latencies) / len(latencies),
        "latency_measurement": "sum of cached B1 and selected A3 component latency; not fresh end-to-end wall clock",
    }, after


def comparison(gold, before, after, samples, seed):
    return {
        "mcnemar_exact": mcnemar_exact(gold, before, after),
        "paired_bootstrap": paired_bootstrap(gold, before, after, samples, seed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--action", default="conflicting_rescue_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    gold_rows = indexed(read_jsonl(args.gold), "gold")
    baseline = indexed(read_jsonl(args.baseline), "baseline")
    candidate = indexed(read_jsonl(args.candidate), "candidate")
    if set(gold_rows) != set(baseline) or set(gold_rows) != set(candidate):
        raise ValueError("Gold, baseline, and candidate IDs must match exactly")
    ids = sorted(gold_rows)
    routes, scores = build_routes(
        ids, baseline, ResidualMemoryStore.load(args.memory), args.action,
        args.top_k, args.min_support, args.cost_weight, args.utility_threshold,
    )
    # Route selection above has no access to gold_rows. Gold is opened only for evaluation.
    results, predictions = {}, {}
    for policy, selected in routes.items():
        result, output = evaluate(ids, gold_rows, baseline, candidate, selected)
        results[policy] = result
        predictions[policy] = output
    expected = [gold_rows[key]["label"] for key in ids]
    comparisons = {
        "R6_vs_B1": comparison(expected, predictions["B1"], predictions["R6"], args.samples, args.seed),
        "A3-All_vs_B1": comparison(expected, predictions["B1"], predictions["A3-All"], args.samples, args.seed),
        "R6_vs_R-Confidence": comparison(expected, predictions["R-Confidence"], predictions["R6"], args.samples, args.seed),
        "R6_vs_R-Similarity": comparison(expected, predictions["R-Similarity"], predictions["R6"], args.samples, args.seed),
    }
    rows = []
    for policy, selected in routes.items():
        for key, prediction in zip(ids, predictions[policy]):
            rows.append({
                "example_id": key, "policy": policy, "routed": key in selected,
                "prediction": prediction, "gold_label": gold_rows[key]["label"],
            })
    summary = {
        "evaluation": "locked_external_four_label_source_only_matched_route_count",
        "n": len(ids), "action": args.action,
        "configuration": {
            "top_k": args.top_k, "min_support": args.min_support,
            "cost_weight": args.cost_weight, "utility_threshold": args.utility_threshold,
            "seed": args.seed, "confidence_and_similarity_matched_to_r6_route_count": True,
            "target_labels_used_for_routing": False,
        },
        "results": results, "comparisons": comparisons,
        "input_artifacts": {
            "gold": {"path": str(args.gold), "sha256": sha256_file(args.gold)},
            "baseline": {"path": str(args.baseline), "sha256": sha256_file(args.baseline)},
            "candidate": {"path": str(args.candidate), "sha256": sha256_file(args.candidate)},
            "memory": {"path": str(args.memory), "sha256": sha256_file(args.memory)},
        },
        "limitations": [
            "Single official labeled release; no target development or retuning is permitted.",
            "Evidence consists of official retrieved-and-annotated candidates, not ERRORFlow retrieval.",
            "Cached component latency is not fresh end-to-end wall-clock latency.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    write_jsonl(args.output_dir / "routes.jsonl", (
        {"policy": policy, "selected_ids": sorted(selected)} for policy, selected in routes.items()
    ))
    print(json.dumps({"results": results, "comparisons": comparisons}, indent=2))
    print(f"EXTERNAL_TRANSFER_DONE rows={len(ids)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
