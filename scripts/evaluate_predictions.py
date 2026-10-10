"""Evaluate prediction JSONL against prepared FEVER JSONL."""

import argparse
import json
from collections import Counter
from pathlib import Path


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gold = {str(json.loads(line)["claim_id"]): json.loads(line)["gold_label"] for line in args.gold.open(encoding="utf-8") if line.strip()}
    predictions = {str(json.loads(line)["claim_id"]): json.loads(line) for line in args.predictions.open(encoding="utf-8") if line.strip()}
    ids = sorted(set(gold) & set(predictions))
    labels = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO")
    confusion = Counter((gold[i], predictions[i]["prediction"]) for i in ids)
    per_class = {}
    for label in labels:
        tp = confusion[(label, label)]
        fp = sum(confusion[(other, label)] for other in labels if other != label)
        fn = sum(confusion[(label, other)] for other in labels if other != label)
        per_class[label] = {"f1": f1(tp, fp, fn), "support": sum(confusion[(label, other)] for other in labels)}
    result = {
        "n_gold": len(gold),
        "n_predictions": len(predictions),
        "n_paired": len(ids),
        "missing_predictions": len(set(gold) - set(predictions)),
        "accuracy": sum(predictions[i]["prediction"] == gold[i] for i in ids) / len(ids) if ids else 0.0,
        "macro_f1": sum(x["f1"] for x in per_class.values()) / len(labels),
        "per_class": per_class,
        "confusion": {f"{a}->{b}": confusion[(a, b)] for a in labels for b in labels},
        "avg_latency_ms": sum(predictions[i].get("latency_ms", 0.0) for i in ids) / len(ids) if ids else 0.0,
        "avg_generated_tokens": sum(predictions[i].get("token_count", 0) for i in ids) / len(ids) if ids else 0.0,
        "avg_llm_calls": sum(predictions[i].get("metadata", {}).get("llm_calls", 1) for i in ids) / len(ids) if ids else 0.0,
        "route_rate": sum(bool(predictions[i].get("metadata", {}).get("route_applied", False)) for i in ids) / len(ids) if ids else 0.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
