"""Build a margin-spanning, transition-stratified V3 train action screen."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_folds import reject_opened_holdout


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def spread_sample(rows, limit, seed, group):
    ordered = sorted(rows, key=lambda row: (
        float(row.get("prediction_margin") or 0.0),
        hashlib.sha256(f"{seed}:{group}:{row['example_id']}".encode()).hexdigest(),
    ))
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    indexes = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    return [ordered[index] for index in indexes]


def build_screen(gold_rows, baseline_rows, per_error_pair=10, controls_per_label=10, seed=2026):
    gold, baseline = indexed(gold_rows, "gold"), indexed(baseline_rows, "baseline")
    if set(gold) != set(baseline):
        raise ValueError("Gold and baseline IDs must match exactly")
    reject_opened_holdout(gold.values())
    groups = defaultdict(list)
    for key, row in baseline.items():
        group = f"control:{row['gold_label']}" if row["correct"] else f"error:{row['gold_label']}->{row['prediction']}"
        groups[group].append(row)
    selected, manifest_groups = [], {}
    for group in sorted(groups):
        limit = controls_per_label if group.startswith("control:") else per_error_pair
        chosen = spread_sample(groups[group], limit, seed, group)
        selected.extend(chosen)
        margins = [float(row.get("prediction_margin") or 0.0) for row in chosen]
        manifest_groups[group] = {
            "population_n": len(groups[group]), "selected_n": len(chosen),
            "selected_margin_min": min(margins), "selected_margin_max": max(margins),
        }
    selected.sort(key=lambda row: row["example_id"])
    examples = []
    for row in selected:
        example = dict(gold[row["example_id"]])
        example["metadata"] = dict(example.get("metadata", {}))
        example["metadata"]["v3_screen"] = {
            "method": "transition_stratified_margin_spread", "seed": seed,
            "baseline_prediction": row["prediction"], "prediction_margin": row.get("prediction_margin"),
        }
        examples.append(example)
    return examples, selected, {
        "method": "transition_stratified_margin_spread", "seed": seed,
        "per_error_pair": per_error_pair, "controls_per_label": controls_per_label,
        "rows": len(selected), "groups": manifest_groups,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-error-pair", type=int, default=10)
    parser.add_argument("--controls-per-label", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    examples, baseline, manifest = build_screen(
        read_jsonl(args.gold), read_jsonl(args.baseline_records),
        args.per_error_pair, args.controls_per_label, args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "screen_examples.jsonl", examples)
    write_jsonl(args.output_dir / "screen_baseline_records.jsonl", baseline)
    atomic_write_json(args.output_dir / "screen_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print(f"V3_SCREEN_DONE rows={len(examples)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
