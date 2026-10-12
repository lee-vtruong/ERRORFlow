"""Build traceable B1 residual records and descriptive train-only metrics."""

import argparse
import json
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_folds import assign_grouped_folds, reject_opened_holdout
from errorflow.v3_residuals import build_residual_records, summarize_residuals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gold = list(read_jsonl(args.gold))
    predictions = list(read_jsonl(args.predictions))
    if args.max_samples is not None:
        gold, predictions = gold[:args.max_samples], predictions[:args.max_samples]
    reject_opened_holdout(gold)
    records = build_residual_records(gold, predictions)
    folds = assign_grouped_folds(gold, args.folds, args.seed)
    for row in records:
        row["outer_fold"] = folds[row["example_id"]]
    summary = summarize_residuals(records)
    summary.update({"folds": args.folds, "seed": args.seed, "development_split": "train_only",
                    "opened_holdout_excluded": True, "dry_run": args.dry_run})
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "baseline_scores.jsonl", records)
    write_jsonl(args.output_dir / "residual_errors.jsonl", (row for row in records if not row["correct"]))
    atomic_write_json(args.output_dir / "residual_metrics.json", summary)
    print(f"V3_RESIDUALS_DONE rows={len(records)} errors={summary['n_errors']} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
