"""Stratify frozen policy results by SciFact retrieval success."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl
from scripts.evaluate_scifact_retrieval import gold_groups


def retrieval_group(claim, retrieved):
    groups = gold_groups(claim)
    if not groups:
        return "nei_no_gold_rationale"
    selected = {
        (str(item.get("metadata", {}).get("doc_id")), int(item.get("metadata", {}).get("sentence_id")))
        for item in retrieved.get("evidence", [])
    }
    return "verifiable_hit" if any(group & selected for group in groups) else "verifiable_miss"


def summarize(claim_rows, retrieved_rows, prediction_rows):
    claims = {str(row["id"]): row for row in claim_rows}
    retrieved = {str(row["example_id"]).split(":", 1)[-1]: row for row in retrieved_rows}
    predictions = defaultdict(dict)
    for row in prediction_rows:
        predictions[row["policy"]][str(row["example_id"]).split(":", 1)[-1]] = row
    if "B1" not in predictions:
        raise ValueError("Prediction artifact must contain B1")
    expected_ids = set(claims)
    for policy, rows in predictions.items():
        if set(rows) != expected_ids:
            raise ValueError(f"Policy {policy} IDs do not match claims")
    grouped_ids = defaultdict(list)
    for claim_id, claim in claims.items():
        if claim_id not in retrieved:
            raise ValueError(f"Missing retrieved row for claim {claim_id}")
        grouped_ids[retrieval_group(claim, retrieved[claim_id])].append(claim_id)

    result = {"groups": {}, "policies": sorted(predictions)}
    for group, ids in sorted(grouped_ids.items()):
        group_result = {"n": len(ids), "systems": {}}
        baseline_rows = predictions["B1"]
        baseline_correct = {
            claim_id: baseline_rows[claim_id]["prediction"] == baseline_rows[claim_id]["gold_label"]
            for claim_id in ids
        }
        for policy, rows in sorted(predictions.items()):
            gold = [rows[claim_id]["gold_label"] for claim_id in ids]
            predicted = [rows[claim_id]["prediction"] for claim_id in ids]
            helpful = sum(
                not baseline_correct[claim_id]
                and rows[claim_id]["prediction"] == rows[claim_id]["gold_label"]
                for claim_id in ids
            )
            harmful = sum(
                baseline_correct[claim_id]
                and rows[claim_id]["prediction"] != rows[claim_id]["gold_label"]
                for claim_id in ids
            )
            group_result["systems"][policy] = {
                "accuracy": accuracy(predicted, gold), "macro_f1": macro_f1(predicted, gold),
                "routed": sum(bool(rows[claim_id].get("routed")) for claim_id in ids),
                "route_rate": sum(bool(rows[claim_id].get("routed")) for claim_id in ids) / len(ids),
                "helpful_vs_b1": helpful, "harmful_vs_b1": harmful, "net_vs_b1": helpful - harmful,
            }
        result["groups"][group] = group_result
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--retrieved", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(
        read_jsonl(args.claims), read_jsonl(args.retrieved), read_jsonl(args.predictions)
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
