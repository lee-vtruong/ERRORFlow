"""Create a deterministic train-only intervention screening set."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl, write_jsonl


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def build_screen(gold_rows, prediction_rows, per_error_pair=10, controls_per_label=10, seed=2026):
    gold, predictions = indexed(gold_rows, "gold"), indexed(prediction_rows, "predictions")
    if set(gold) - set(predictions):
        raise ValueError("Baseline predictions are missing gold IDs")
    errors, controls = defaultdict(list), defaultdict(list)
    for key, row in gold.items():
        expected, predicted = row["label"], predictions[key]["prediction"]
        bucket = f"{expected}->{predicted}"
        (controls[expected] if expected == predicted else errors[bucket]).append(key)
    selected = []
    counts = {"errors": {}, "controls": {}}
    for role, groups, limit in (("error", errors, per_error_pair), ("control", controls, controls_per_label)):
        for group in sorted(groups):
            ids = sorted(groups[group])
            random.Random(f"{seed}:{role}:{group}").shuffle(ids)
            chosen = ids[:limit]
            counts["errors" if role == "error" else "controls"][group] = len(chosen)
            for key in chosen:
                row = dict(gold[key])
                row["metadata"] = dict(row.get("metadata", {}))
                row["metadata"].update({"screen_role": role, "screen_group": group, "baseline_prediction": predictions[key]["prediction"]})
                selected.append(row)
    selected.sort(key=lambda row: row["example_id"])
    return selected, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-error-pair", type=int, default=10)
    parser.add_argument("--controls-per-label", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rows, counts = build_screen(read_jsonl(args.gold), read_jsonl(args.baseline), args.per_error_pair, args.controls_per_label, args.seed)
    write_jsonl(args.output, rows)
    manifest = {"seed": args.seed, "per_error_pair": args.per_error_pair,
                "controls_per_label": args.controls_per_label, "rows": len(rows), **counts}
    atomic_write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))
    print(f"SCREEN_READY rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
