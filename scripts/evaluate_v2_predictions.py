"""Evaluate canonical V2 predictions with strict ID pairing."""

import argparse
import json
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    gold, predictions = indexed(read_jsonl(args.gold), "gold"), indexed(read_jsonl(args.predictions), "predictions")
    missing, extra = sorted(set(gold) - set(predictions)), sorted(set(predictions) - set(gold))
    paired = sorted(set(gold) & set(predictions))
    gold_labels = [gold[key]["label"] for key in paired]
    predicted_labels = [predictions[key]["prediction"] for key in paired]
    labels = sorted(set(gold_labels) | set(predicted_labels))
    confusion = {f"{g}->{p}": sum(gg == g and pp == p for gg, pp in zip(gold_labels, predicted_labels))
                 for g in labels for p in labels}
    metrics = {"n_gold": len(gold), "n_predictions": len(predictions), "n_paired": len(paired),
               "missing_predictions": len(missing), "extra_predictions": len(extra),
               "accuracy": accuracy(predicted_labels, gold_labels), "macro_f1": macro_f1(predicted_labels, gold_labels),
               "labels": labels, "confusion": confusion,
               "avg_latency_ms": sum(float(predictions[k].get("latency_ms", 0)) for k in paired) / len(paired) if paired else 0.0,
               "avg_generated_tokens": sum(float(predictions[k].get("token_count", 0)) for k in paired) / len(paired) if paired else 0.0}
    atomic_write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
