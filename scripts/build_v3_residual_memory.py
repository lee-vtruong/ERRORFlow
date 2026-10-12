"""Build train-only V3 action-outcome memory from cached counterfactuals."""

import argparse
import json
from pathlib import Path

from errorflow.io import read_jsonl
from errorflow.v3_memory import ResidualMemoryStore, build_action_memories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-records", required=True, type=Path)
    parser.add_argument("--candidate", nargs="+", required=True, help="ACTION=predictions.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = list(read_jsonl(args.residual_records))
    if args.max_samples is not None:
        rows = rows[:args.max_samples]
    named = []
    for specification in args.candidate:
        action, separator, filename = specification.partition("=")
        if not separator:
            raise ValueError(f"Expected ACTION=PATH, got {specification}")
        candidate_rows = list(read_jsonl(filename))
        if args.max_samples is not None:
            allowed = {row["example_id"] for row in rows}
            candidate_rows = [row for row in candidate_rows if row["example_id"] in allowed]
        named.append((action, candidate_rows))
    records = build_action_memories(rows, named)
    summary = {
        "records": len(records), "examples": len({record.source_claim_id for record in records}),
        "actions": sorted({record.correction_strategy for record in records}),
        "recovered": sum(record.recovery_count for record in records),
        "regressed": sum(record.regression_count for record in records),
        "train_only": all(record.source_split == "train" for record in records),
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2))
    if not args.dry_run:
        ResidualMemoryStore(records).save(args.output)
        print(f"V3_MEMORY_DONE records={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
