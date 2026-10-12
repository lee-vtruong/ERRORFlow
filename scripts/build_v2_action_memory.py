"""Build train-only action-outcome memory, including harmful controls."""

import argparse
import json
from pathlib import Path

from errorflow.error_memory import ErrorMemoryStore, memory_id
from errorflow.io import read_jsonl
from errorflow.schemas import ErrorMemoryRecord


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def build_records(gold_rows, baseline_rows, named_candidates, dataset="AVeriTeC", source_split="train"):
    gold, baseline = indexed(gold_rows, "gold"), indexed(baseline_rows, "baseline")
    candidates = {action: indexed(rows, action) for action, rows in named_candidates}
    records = []
    for key, row in gold.items():
        before = baseline[key]
        expected = row["label"]
        before_label = before["prediction"]
        before_correct = before_label == expected
        group = f"control:{expected}" if before_correct else f"error:{expected}->{before_label}"
        for action, action_rows in candidates.items():
            after = action_rows[key]
            after_correct = after["prediction"] == expected
            recovered = not before_correct and after_correct
            regressed = before_correct and not after_correct
            utility = 1.0 if recovered else (-1.0 if regressed else 0.0)
            records.append(ErrorMemoryRecord(
                memory_id=memory_id(dataset, key, group, action), dataset=dataset,
                source_split=source_split, claim_id=key, claim=row["claim"],
                error_type=group, baseline_prediction=before_label,
                diagnosis=group, action=action, gold_label=expected,
                final_prediction=after["prediction"], recovered=recovered,
                regressed=regressed, utility=utility,
                extra_tokens=int(after.get("token_count", 0)), extra_llm_calls=0,
                latency_ms=float(after.get("latency_ms", 0)), allow_for_routing=True,
                metadata={
                    "record_role": "control" if before_correct else "error",
                    "baseline_was_correct": before_correct,
                    "baseline_confidence": before.get("confidence"),
                    "baseline_label_scores": before.get("metadata", {}).get("label_scores", {}),
                    "candidate_label_scores": after.get("metadata", {}).get("label_scores", {}),
                    "candidate_prediction": after["prediction"],
                    "evidence_count": len(row.get("evidence", [])),
                    "model_forward_passes": after.get("metadata", {}).get("model_forward_passes", 1),
                },
            ))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", nargs="+", required=True, help="ACTION=predictions.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    named = []
    for specification in args.candidate:
        action, separator, filename = specification.partition("=")
        if not separator:
            raise ValueError(f"Expected ACTION=PATH, got {specification}")
        named.append((action, read_jsonl(filename)))
    records = build_records(read_jsonl(args.gold), read_jsonl(args.baseline), named)
    ErrorMemoryStore(records).save(args.output)
    summary = {
        "records": len(records), "examples": len({record.claim_id for record in records}),
        "actions": sorted({record.action for record in records}),
        "recovered": sum(record.recovered for record in records),
        "regressed": sum(record.regressed for record in records),
        "positive_utility": sum(record.utility > 0 for record in records),
        "negative_utility": sum(record.utility < 0 for record in records),
    }
    print(json.dumps(summary, indent=2))
    print(f"ACTION_MEMORY_READY output={args.output}")


if __name__ == "__main__":
    main()
