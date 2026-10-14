"""Evaluate NE6 source-memory subsets on a locked external target dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_memory import ResidualMemoryStore, record_from_dict
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap

try:
    from scripts.evaluate_v3_nlf_policy import (
        evaluate, gold_label, indexed, route_scores, sha256_file,
    )
except ModuleNotFoundError:
    from evaluate_v3_nlf_policy import evaluate, gold_label, indexed, route_scores, sha256_file


def subset_definitions(records):
    definitions = {"full": records}
    predicted = defaultdict(list)
    source_gold = defaultdict(list)
    topics = defaultdict(list)
    for record in records:
        label = record.retrieval_features.get("baseline_prediction")
        if label is not None:
            predicted[str(label)].append(record)
        gold = record.provenance.get("gold_label_offline_only")
        if gold is not None:
            source_gold[str(gold)].append(record)
        topic = record.provenance.get("topic") or record.retrieval_features.get("topic")
        if topic is not None:
            topics[str(topic)].append(record)
    for label, rows in sorted(predicted.items()):
        definitions[f"source_predicted_label={label}"] = rows
    for label, rows in sorted(source_gold.items()):
        definitions[f"source_gold_label={label}"] = rows
    for topic, rows in sorted(topics.items()):
        definitions[f"source_topic={topic}"] = rows
    return definitions, bool(topics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--action", default="conflicting_rescue_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--project-conflicting-to-nei", action="store_true")
    args = parser.parse_args()

    baseline = indexed(read_jsonl(args.baseline), "baseline")
    candidate = indexed(read_jsonl(args.candidate), "candidate")
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate IDs must match exactly")
    ids = sorted(baseline)
    records = [record_from_dict(row) for row in read_jsonl(args.memory)]
    subsets, topics_available = subset_definitions(records)
    projection = {"CONFLICTING": "NOT ENOUGH INFO"} if args.project_conflicting_to_nei else {}
    results, outputs, routes, prediction_rows, overhead = {}, {}, {}, [], {}
    route_materialization = {}

    # Materialize every route before opening the target gold artifact.
    for name, subset in subsets.items():
        print(f"NE6 subset={name} records={len(subset)}", flush=True)
        scores, selected, _, timing = route_scores(
            ids, baseline, ResidualMemoryStore(subset), args.action,
            args.top_k, args.min_support, args.cost_weight,
            args.utility_threshold, "none", grouped=False,
        )
        route_materialization[name] = (scores, selected, len(subset), timing)

    gold_rows = indexed(read_jsonl(args.gold), "gold")
    if set(gold_rows) != set(baseline):
        raise ValueError("Gold, baseline, and candidate IDs must match exactly")
    expected = [gold_label(gold_rows[key]) for key in ids]
    for name, (scores, selected, subset_size, timing) in route_materialization.items():
        metrics, predictions = evaluate(ids, gold_rows, baseline, candidate, selected, projection)
        metrics["source_memory_records"] = subset_size
        results[name] = metrics
        outputs[name] = predictions
        routes[name] = selected
        overhead[name] = timing
        for key, prediction in zip(ids, predictions):
            prediction_rows.append({
                "example_id": key, "memory_subset": name,
                "source_memory_records": subset_size, "routed": key in selected,
                "prediction": prediction, "gold_label": gold_label(gold_rows[key]),
                "estimated_utility": scores[key] if scores[key] != float("-inf") else None,
            })

    comparisons = {}
    full = outputs["full"]
    for name, predictions in outputs.items():
        if name == "full":
            continue
        comparisons[name] = {
            "orientation": "full_minus_subset",
            "mcnemar_exact": mcnemar_exact(expected, predictions, full),
            "paired_bootstrap_full_minus_subset": paired_bootstrap(
                expected, predictions, full, args.samples, args.seed,
            ),
        }

    summary = {
        "evaluation": "locked_external_ne6_source_memory_sensitivity",
        "configuration": {
            "action": args.action, "compatibility": "none",
            "top_k": args.top_k, "min_support": args.min_support,
            "cost_weight": args.cost_weight,
            "utility_threshold": args.utility_threshold,
            "seed": args.seed, "target_labels_used_for_routing": False,
            "evaluation_label_projection": projection,
        },
        "results": results, "comparisons": comparisons,
        "retrieval_overhead": overhead,
        "topic_subsets_available": topics_available,
        "input_artifacts": {
            "gold": {"path": str(args.gold), "sha256": sha256_file(args.gold)},
            "baseline": {"path": str(args.baseline), "sha256": sha256_file(args.baseline)},
            "candidate": {"path": str(args.candidate), "sha256": sha256_file(args.candidate)},
            "memory": {"path": str(args.memory), "sha256": sha256_file(args.memory)},
        },
        "protocol_notes": [
            "Every subset is constructed solely from source-memory metadata.",
            "Target labels are read only after every route has been materialized.",
            "Predicted-label and source-gold-label subsets are both reported; they answer different composition questions.",
            "Topic subsets are omitted rather than fabricated when source memory has no topic metadata.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(args.output_dir / "routes.jsonl", (
        {"memory_subset": name, "selected_ids": sorted(selected)}
        for name, selected in routes.items()
    ))
    print(json.dumps({
        "subsets": len(subsets), "topics_available": topics_available,
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
