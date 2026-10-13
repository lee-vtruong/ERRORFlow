"""Evaluate SciFact sentence retrieval against official dev rationales."""

import argparse
import json
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl


def gold_groups(claim):
    return [
        {(str(doc_id), int(sentence_id)) for sentence_id in rationale.get("sentences", [])}
        for doc_id, rationale_sets in (claim.get("evidence") or {}).items()
        for rationale in rationale_sets
        if rationale.get("sentences")
    ]


def retrieval_metrics(claim_rows, retrieved_rows, ks):
    claims = {str(row["id"]): row for row in claim_rows}
    retrieved = {str(row["example_id"]).split(":", 1)[-1]: row for row in retrieved_rows}
    paired = []
    for claim_id, claim in claims.items():
        groups = gold_groups(claim)
        if not groups or claim_id not in retrieved:
            continue
        ranked = [
            (str(item.get("metadata", {}).get("doc_id")), int(item.get("metadata", {}).get("sentence_id")))
            for item in retrieved[claim_id].get("evidence", [])
        ]
        paired.append((groups, ranked))
    metrics = {}
    for k in ks:
        any_hits, complete_hits, coverage = 0, 0, []
        for groups, ranked in paired:
            selected = set(ranked[:k])
            group_coverages = [len(group & selected) / len(group) for group in groups]
            best = max(group_coverages, default=0.0)
            any_hits += best > 0
            complete_hits += best == 1.0
            coverage.append(best)
        metrics[str(k)] = {
            "claim_any_sentence_recall": any_hits / len(paired) if paired else 0.0,
            "complete_rationale_recall": complete_hits / len(paired) if paired else 0.0,
            "mean_best_rationale_coverage": sum(coverage) / len(coverage) if coverage else 0.0,
        }
    return {
        "n_claims": len(claims), "n_retrieved": len(retrieved),
        "n_verifiable_paired": len(paired), "metrics_at_k": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--retrieved", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5])
    args = parser.parse_args()
    result = retrieval_metrics(read_jsonl(args.claims), read_jsonl(args.retrieved), args.ks)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
