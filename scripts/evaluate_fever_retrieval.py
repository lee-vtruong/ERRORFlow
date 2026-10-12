"""Evaluate FEVER retrieval against annotated evidence provenance."""
import argparse
import json
from pathlib import Path


def gold_evidence_groups(row):
    groups = []
    for raw_group in row.get("evidence") or []:
        group = set()
        for item in raw_group or []:
            if len(item) >= 4 and item[2] is not None and item[3] is not None:
                group.add((str(item[2]), int(item[3])))
        if group:
            groups.append(group)
    return groups


def retrieved_evidence(row):
    return [
        (str(item["page_id"]), int(item["sentence_id"]))
        for item in row.get("evidence_metadata") or []
    ]


def retrieval_metrics(gold_rows, retrieved_rows, ks):
    retrieved_by_id = {str(row["claim_id"]): row for row in retrieved_rows}
    gold_ids = {str(row.get("id", row.get("claim_id"))) for row in gold_rows}
    paired = []
    missing_metadata = 0
    no_gold_evidence = 0

    for gold in gold_rows:
        claim_id = str(gold.get("id", gold.get("claim_id")))
        retrieved = retrieved_by_id.get(claim_id)
        if retrieved is None:
            continue
        if "evidence_metadata" not in retrieved:
            missing_metadata += 1
            continue
        groups = gold_evidence_groups(gold)
        if not groups:
            no_gold_evidence += 1
            continue
        paired.append((groups, retrieved_evidence(retrieved)))

    if missing_metadata:
        raise ValueError(
            f"{missing_metadata} paired retrieval rows lack evidence_metadata; "
            "rerun retrieval with provenance-enabled retrieve_fts.py"
        )

    metrics = {}
    denominator = len(paired)
    for k in sorted(set(ks)):
        any_hits = complete_hits = 0
        best_coverages = []
        for groups, ranked in paired:
            top_k = set(ranked[:k])
            coverages = [len(group & top_k) / len(group) for group in groups]
            best = max(coverages)
            any_hits += best > 0
            complete_hits += best == 1.0
            best_coverages.append(best)
        metrics[str(k)] = {
            "claim_any_sentence_recall": any_hits / denominator if denominator else 0.0,
            "complete_evidence_group_recall": complete_hits / denominator if denominator else 0.0,
            "mean_best_group_coverage": sum(best_coverages) / denominator if denominator else 0.0,
        }

    return {
        "n_gold": len(gold_rows),
        "n_retrieved": len(retrieved_rows),
        "n_paired_verifiable": denominator,
        "n_paired_without_gold_evidence": no_gold_evidence,
        "retrieved_ids_missing_from_gold": sum(claim_id not in gold_ids for claim_id in retrieved_by_id),
        "metrics_at_k": metrics,
    }


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--retrieved", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    args = parser.parse_args()
    if any(k <= 0 for k in args.ks):
        parser.error("--ks values must be positive")
    result = retrieval_metrics(read_jsonl(args.gold), read_jsonl(args.retrieved), args.ks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
