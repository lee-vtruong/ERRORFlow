"""Export one genuine, fold-safe R6 decision trace from server artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from errorflow.io import read_jsonl
from errorflow.v3_memory import ResidualMemoryStore
from errorflow.v3_routing import estimate_action


def index(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--action", default="conflicting_rescue_v1")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefer", choices=("helpful", "harmful", "any"), default="helpful")
    args = parser.parse_args()

    gold = index(read_jsonl(args.gold), "gold")
    baseline = index(read_jsonl(args.baseline_records), "baseline")
    candidate = index(read_jsonl(args.candidate), "candidate")
    policy = index(read_jsonl(args.policy), "policy")
    store = ResidualMemoryStore.load(args.memory)

    eligible = []
    for key, route in policy.items():
        if not route.get("routed"):
            continue
        b, a, g = baseline[key], candidate[key], gold[key]
        helpful = b["prediction"] != g["label"] and a["prediction"] == g["label"]
        harmful = b["prediction"] == g["label"] and a["prediction"] != g["label"]
        if args.prefer == "helpful" and not helpful:
            continue
        if args.prefer == "harmful" and not harmful:
            continue
        eligible.append((float(route.get("estimated_utility") or float("-inf")), key))
    if not eligible:
        raise ValueError(f"no routed {args.prefer} example found")
    _, key = max(eligible)
    b, a, g = baseline[key], candidate[key], gold[key]
    query = {"claim": b["claim"], "retrieval_features": {
        "baseline_prediction": b["prediction"],
        "label_scores": b.get("label_scores", {}),
        "evidence_tokens": b.get("evidence_tokens", 0),
    }}
    fold = b.get("outer_fold")
    neighbors = store.retrieve(
        query, top_k=5, action=args.action, exclude_claim_ids={key},
        exclude_folds={fold} if fold is not None else None,
    )
    estimate = estimate_action(args.action, neighbors, cost_weight=0.1)
    payload = {
        "example_id": key,
        "claim": g["claim"],
        "evidence": g.get("evidence", []),
        "gold_label_offline_for_audit_only": g["label"],
        "b1_prediction": b["prediction"],
        "a3_prediction": a["prediction"],
        "routed": estimate.utility > 0 and estimate.support >= 2,
        "estimate": estimate.to_dict(),
        "neighbors": [{
            "rank": rank, "memory_id": record.memory_id,
            "source_claim_id": record.source_claim_id,
            "source_claim": record.source_claim, "similarity": similarity,
            "recovery": record.recovery_count,
            "regression": record.regression_count,
            "normalized_cost": record.estimated_extra_cost,
        } for rank, (record, similarity) in enumerate(neighbors, 1)],
        "leakage_guards": {
            "query_id_excluded": True,
            "query_fold_excluded": fold is not None,
            "gold_not_used_for_routing": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WORKED_EXAMPLE_DONE example_id={key} output={args.output}")


if __name__ == "__main__":
    main()
