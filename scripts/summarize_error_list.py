"""Summarize observable baseline errors, confidence, and evidence sources."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize(rows):
    pairs = defaultdict(lambda: {"n": 0, "confidence_sum": 0.0, "confidence_n": 0})
    top1_sources = Counter()
    evidence_sources = Counter()
    gold_errors = Counter()
    predicted_errors = Counter()
    confidence_bins = Counter()

    for row in rows:
        pair = f'{row["gold_label"]}->{row["baseline_prediction"]}'
        group = pairs[pair]
        group["n"] += 1
        confidence = row.get("baseline_confidence")
        if confidence is not None:
            confidence = float(confidence)
            group["confidence_sum"] += confidence
            group["confidence_n"] += 1
            lower = min(int(confidence * 10) / 10, 0.9)
            confidence_bins[f"{lower:.1f}-{lower + 0.1:.1f}"] += 1
        gold_errors[row["gold_label"]] += 1
        predicted_errors[row["baseline_prediction"]] += 1
        metadata = row.get("evidence_metadata") or []
        if metadata:
            top1_sources[metadata[0].get("source", "unknown")] += 1
        for item in metadata:
            evidence_sources[item.get("source", "unknown")] += 1

    pair_output = {}
    for pair, values in sorted(pairs.items(), key=lambda item: (-item[1]["n"], item[0])):
        pair_output[pair] = {
            "n": values["n"],
            "mean_baseline_confidence": (
                values["confidence_sum"] / values["confidence_n"]
                if values["confidence_n"] else None
            ),
        }
    return {
        "n_errors": len(rows),
        "label_pairs": pair_output,
        "errors_by_gold_label": dict(gold_errors),
        "errors_by_predicted_label": dict(predicted_errors),
        "confidence_bins": dict(sorted(confidence_bins.items())),
        "top1_evidence_source": dict(top1_sources),
        "all_evidence_sources": dict(evidence_sources),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    result = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
