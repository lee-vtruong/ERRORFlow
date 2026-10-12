"""Materialize fixed R6 predictions from cached actions and fold-safe memory."""

import argparse
import json
import time
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_memory import ResidualMemoryStore
from errorflow.v3_routing import best_memory_action


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def materialize(baseline_rows, candidate_rows, store, action, top_k=5,
                min_support=2, cost_weight=0.1, utility_threshold=0.0,
                progress_every=50):
    baseline = indexed(baseline_rows, "baseline")
    candidate = indexed(candidate_rows, "candidate")
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate IDs must match exactly")
    ids, output = sorted(baseline), []
    started = time.perf_counter()
    for index, key in enumerate(ids, 1):
        row = baseline[key]
        query = {"claim": row["claim"], "retrieval_features": {
            "baseline_prediction": row["prediction"],
            "label_scores": row.get("label_scores", {}),
            "evidence_tokens": row.get("evidence_tokens", 0),
        }}
        neighbors = store.retrieve(
            query, top_k=top_k, action=action, exclude_claim_ids={key},
            exclude_folds={row.get("outer_fold")},
        )
        best, estimates = best_memory_action(
            [action], neighbors, cost_weight=cost_weight,
            min_support=min_support, utility_aware=True,
        )
        routed = best is not None and best.utility > utility_threshold
        final_prediction = candidate[key]["prediction"] if routed else row["prediction"]
        output.append({
            "example_id": key, "prediction": final_prediction,
            "baseline_prediction": row["prediction"], "gold_label": row["gold_label"],
            "routed": routed, "action": action if routed else "none",
            "estimated_utility": best.utility if best else None,
            "estimated_support": best.support if best else 0,
            "latency_ms": float(row.get("latency_ms", 0.0)) +
                          (float(candidate[key].get("latency_ms", 0.0)) if routed else 0.0),
            "token_count": int(row.get("token_count", 0)) +
                           (int(candidate[key].get("token_count", 0)) if routed else 0),
            "metadata": {
                "policy": "R6_utility", "outer_fold": row.get("outer_fold"),
                "memory_fold_excluded": True, "utility_threshold": utility_threshold,
                "top_k": top_k, "min_support": min_support, "cost_weight": cost_weight,
                "model_forward_passes": 4 + 4 * int(routed),
                "llm_requests": 1 + int(routed),
                "action_estimates": [estimate.to_dict() for estimate in estimates],
            },
        })
        if progress_every and (index % progress_every == 0 or index == len(ids)):
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0.0
            eta = (len(ids) - index) / rate if rate else 0.0
            print(f"materialize={index}/{len(ids)} rate={rate:.2f}/s eta={eta:.1f}s", flush=True)
    return output


def summarize(rows):
    gold = [row["gold_label"] for row in rows]
    baseline = [row["baseline_prediction"] for row in rows]
    candidate = [row["prediction"] for row in rows]
    helpful = sum(b != g and c == g for g, b, c in zip(gold, baseline, candidate))
    harmful = sum(b == g and c != g for g, b, c in zip(gold, baseline, candidate))
    routed = sum(row["routed"] for row in rows)
    return {
        "n": len(rows), "policy": "R6_utility", "routed": routed,
        "route_rate": routed / len(rows), "helpful": helpful, "harmful": harmful,
        "net": helpful - harmful,
        "baseline_accuracy": accuracy(baseline, gold), "candidate_accuracy": accuracy(candidate, gold),
        "delta_accuracy": accuracy(candidate, gold) - accuracy(baseline, gold),
        "baseline_macro_f1": macro_f1(baseline, gold), "candidate_macro_f1": macro_f1(candidate, gold),
        "delta_macro_f1": macro_f1(candidate, gold) - macro_f1(baseline, gold),
        "avg_model_forward_passes": sum(row["metadata"]["model_forward_passes"] for row in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--action", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    args = parser.parse_args()
    rows = materialize(
        read_jsonl(args.baseline_records), read_jsonl(args.candidate),
        ResidualMemoryStore.load(args.memory), args.action, args.top_k,
        args.min_support, args.cost_weight, args.utility_threshold,
    )
    metrics = summarize(rows)
    write_jsonl(args.output, rows)
    atomic_write_json(args.metrics_output, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"V3_FIXED_POLICY_DONE rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
