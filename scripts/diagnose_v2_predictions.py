"""Diagnose V2 parsing, truncation, confidence, and label transitions."""

import argparse
import json
from collections import Counter
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl


def index(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def diagnose(gold_rows, prediction_rows, max_new_tokens):
    gold, predictions = index(gold_rows, "gold"), index(prediction_rows, "predictions")
    paired = sorted(set(gold) & set(predictions))
    token_counts = [int(predictions[key].get("token_count", 0)) for key in paired]
    transitions = Counter()
    errors = []
    for key in paired:
        expected, predicted = gold[key]["label"], predictions[key]["prediction"]
        transitions[f"{expected}->{predicted}"] += 1
        if expected != predicted:
            errors.append({
                "example_id": key, "gold_label": expected, "prediction": predicted,
                "token_count": predictions[key].get("token_count"),
                "confidence": predictions[key].get("confidence"),
                "raw_output": predictions[key].get("metadata", {}).get("raw_output", ""),
            })
    return {
        "n_paired": len(paired),
        "prediction_distribution": dict(Counter(predictions[key]["prediction"] for key in paired)),
        "label_transitions": dict(sorted(transitions.items())),
        "at_generation_limit": sum(count >= max_new_tokens for count in token_counts),
        "generation_limit": max_new_tokens,
        "at_generation_limit_rate": sum(count >= max_new_tokens for count in token_counts) / len(paired) if paired else 0.0,
        "confidence_type": dict(Counter(predictions[key].get("metadata", {}).get("confidence_type", "missing") for key in paired)),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    report = diagnose(read_jsonl(args.gold), read_jsonl(args.predictions), args.max_new_tokens)
    atomic_write_json(args.output, report)
    printable = {key: value for key, value in report.items() if key != "errors"}
    printable["n_errors"] = len(report["errors"])
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
