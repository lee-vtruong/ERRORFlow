"""Replay a frozen memory router over precomputed intervention candidates."""

import argparse
import json
from collections import Counter
from pathlib import Path

from errorflow.error_memory import ErrorMemoryStore
from errorflow.io import read_jsonl, write_jsonl
from errorflow.routing_v2 import select_action


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def replay(examples, baseline, candidates, memory, config):
    expected_ids = {str(row["example_id"]) for row in examples}
    if set(baseline) != expected_ids:
        raise ValueError("Baseline IDs do not exactly match input IDs")
    for action, rows in candidates.items():
        if set(rows) != expected_ids:
            raise ValueError(f"Candidate IDs do not exactly match input IDs: {action}")
    outputs = []
    for row in examples:
        key = str(row["example_id"])
        before = baseline[key]
        neighbors = []
        for action in config["actions"]:
            neighbors.extend(memory.retrieve(
                row["claim"], dataset=row.get("dataset"),
                top_k=int(config["top_k_per_action"]),
                allowed_splits=set(config["memory_splits"]),
                baseline_prediction=before["prediction"],
                label_scores=before.get("metadata", {}).get("label_scores"),
                action=action,
            ))
        confidence = before.get("confidence")
        uncertainty = 1.0 if config["uncertainty_mode"] == "full" or confidence is None else 1.0 - float(confidence)
        decision = select_action(
            neighbors, uncertainty=uncertainty,
            cost_weight=float(config["cost_weight"]),
            min_support=int(config["min_support"]),
            min_similarity=float(config["min_similarity"]),
            min_utility=float(config["min_utility"]),
        )
        selected = candidates[decision.action][key] if decision.route else before
        baseline_passes = float(before.get("metadata", {}).get("model_forward_passes", 1))
        extra_passes = float(selected.get("metadata", {}).get("model_forward_passes", 1)) if decision.route else 0.0
        outputs.append({
            "example_id": key,
            "dataset": row.get("dataset"),
            "split": row.get("split"),
            "prediction": selected["prediction"],
            "confidence": selected.get("confidence"),
            "baseline_prediction": before["prediction"],
            "baseline_confidence": before.get("confidence"),
            "routed": decision.route,
            "action": decision.action,
            "latency_ms": float(before.get("latency_ms", 0)) + (float(selected.get("latency_ms", 0)) if decision.route else 0.0),
            "token_count": float(before.get("token_count", 0)) + (float(selected.get("token_count", 0)) if decision.route else 0.0),
            "metadata": {
                "policy_id": config["policy_id"],
                "route": decision.to_dict(),
                "model_forward_passes": baseline_passes + extra_passes,
                "baseline_label_scores": before.get("metadata", {}).get("label_scores", {}),
                "candidate_label_scores": selected.get("metadata", {}).get("label_scores", {}) if decision.route else {},
                "gold_label_accessed_by_router": False,
            },
        })
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--candidate", nargs="+", required=True, help="ACTION=predictions.jsonl")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = {}
    for specification in args.candidate:
        action, separator, filename = specification.partition("=")
        if not separator:
            raise ValueError(f"Expected ACTION=PATH, got {specification}")
        candidates[action] = indexed(read_jsonl(filename), action)
    if set(candidates) != set(config["actions"]):
        raise ValueError("Candidate actions do not match the frozen config")
    outputs = replay(
        list(read_jsonl(args.input)), indexed(read_jsonl(args.baseline), "baseline"),
        candidates, ErrorMemoryStore.load(args.memory), config,
    )
    write_jsonl(args.output, outputs)
    actions = Counter(row["action"] for row in outputs)
    print(f"FROZEN_ROUTER_DONE rows={len(outputs)} routed={len(outputs)-actions['none']} actions={dict(actions)} output={args.output}")


if __name__ == "__main__":
    main()
