"""Isolated grouped evaluation for the frozen Standard Base/Standard Alt pair.

This entry point deliberately refuses historical audit-pair artifacts.  It
validates adjacent generation manifests before delegating to the established
fold-safe in-domain outcome-memory evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl
from errorflow.v3_prompts import V3_ACTION_PROMPTS, prompt_sha256
try:
    from scripts.evaluate_v3_in_domain_memory import main as evaluate_in_domain, sha256_file
except ModuleNotFoundError:
    # Direct execution adds ``scripts/`` rather than the repository root to
    # sys.path.  Keep the module importable both as ``python scripts/...`` and
    # from the test suite.
    from evaluate_v3_in_domain_memory import main as evaluate_in_domain, sha256_file


BASE_ACTION = "standard_verdict_v1"
ALT_ACTION = "standard_deliberate_verdict_v1"
FORBIDDEN_ACTIONS = {"conflict_audit_v1", "conflicting_rescue_v1"}


def sibling_manifest(path: Path) -> Path:
    return path.with_name(path.stem + ".manifest.json")


def load_complete_manifest(predictions: Path, expected_action: str) -> tuple[Path, dict]:
    path = sibling_manifest(predictions)
    if not path.exists():
        raise FileNotFoundError(f"Missing adjacent generation manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    action = config.get("action")
    if action in FORBIDDEN_ACTIONS:
        raise ValueError(f"Historical audit-pair action is forbidden: {action}")
    if action != expected_action:
        raise ValueError(f"Expected action {expected_action}, got {action}")
    expected_hash = prompt_sha256(V3_ACTION_PROMPTS[expected_action])
    if config.get("prompt_sha256") != expected_hash:
        raise ValueError(f"Frozen prompt hash mismatch for {expected_action}")
    if manifest.get("status") != "complete":
        raise ValueError(f"Manifest is not complete: {path}")
    if manifest.get("completed_rows") != manifest.get("requested_rows"):
        raise ValueError(f"Incomplete prediction count in {path}")
    return path, manifest


def validate_pair(base_path: Path, alt_path: Path) -> dict:
    base_manifest_path, base = load_complete_manifest(base_path, BASE_ACTION)
    alt_manifest_path, alt = load_complete_manifest(alt_path, ALT_ACTION)
    base_config, alt_config = base["config"], alt["config"]
    equal_fields = (
        "model", "model_hash", "input", "input_sha256", "split",
        "generation_config_hash", "seed",
    )
    mismatches = {
        field: [base_config.get(field), alt_config.get(field)]
        for field in equal_fields
        if base_config.get(field) != alt_config.get(field)
    }
    if mismatches:
        raise ValueError(f"Base/Alt manifest mismatch: {mismatches}")
    if base_config.get("generation", {}).get("labels") != alt_config.get("generation", {}).get("labels"):
        raise ValueError("Base/Alt label spaces differ")
    if base_config.get("generation", {}).get("labels") != [
        "SUPPORTED", "REFUTED", "NOT ENOUGH INFO", "CONFLICTING",
    ]:
        raise ValueError("The decisive Standard pair requires the frozen four-label protocol")

    base_rows = list(read_jsonl(base_path))
    alt_rows = list(read_jsonl(alt_path))
    base_ids = [str(row["example_id"]) for row in base_rows]
    alt_ids = [str(row["example_id"]) for row in alt_rows]
    if len(base_ids) != len(set(base_ids)) or len(alt_ids) != len(set(alt_ids)):
        raise ValueError("Duplicate prediction IDs")
    if base_ids != alt_ids:
        raise ValueError("Base/Alt prediction ordering or IDs differ")
    for rows, action, manifest in (
        (base_rows, BASE_ACTION, base), (alt_rows, ALT_ACTION, alt),
    ):
        prompt_hash = manifest["config"]["prompt_sha256"]
        for row in rows:
            metadata = row.get("metadata", {})
            if metadata.get("instruction_profile") != action:
                raise ValueError(f"Row action mismatch for {row['example_id']}")
            if metadata.get("prompt_sha256") != prompt_hash:
                raise ValueError(f"Row prompt hash mismatch for {row['example_id']}")
            if metadata.get("gold_label_accessed") is not False:
                raise ValueError(f"Generation accessed gold for {row['example_id']}")
    return {
        "status": "validated",
        "pair": "standard_verdict_v1__standard_deliberate_verdict_v1",
        "historical_audit_memory_allowed": False,
        "base_manifest": str(base_manifest_path),
        "alt_manifest": str(alt_manifest_path),
        "base_config_id": base.get("config_id"),
        "alt_config_id": alt.get("config_id"),
        "model_hash": base_config.get("model_hash"),
        "input_sha256": base_config.get("input_sha256"),
        "generation_config_hash": base_config.get("generation_config_hash"),
        "rows": len(base_rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--standard-base", required=True, type=Path)
    parser.add_argument("--standard-alt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--group-threshold", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=list(range(2026, 2046)))
    args = parser.parse_args(argv)

    provenance = validate_pair(args.standard_base, args.standard_alt)
    gold_hash = sha256_file(args.gold)
    if gold_hash != provenance["input_sha256"]:
        raise ValueError(
            f"Gold input hash does not match generation manifests: {gold_hash} != "
            f"{provenance['input_sha256']}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "standard_pair_provenance.json", provenance)
    delegated = [
        "--gold", str(args.gold),
        "--baseline", str(args.standard_base),
        "--candidate", str(args.standard_alt),
        "--output-dir", str(args.output_dir),
        "--action", ALT_ACTION,
        "--folds", str(args.folds),
        "--group-threshold", str(args.group_threshold),
        "--top-k", str(args.top_k),
        "--min-support", str(args.min_support),
        "--cost-weight", str(args.cost_weight),
        "--samples", str(args.samples),
        "--seed", str(args.seed),
        "--random-seeds", *[str(seed) for seed in args.random_seeds],
    ]
    if args.dataset:
        delegated.extend(["--dataset", args.dataset])
    evaluate_in_domain(delegated)

    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["evaluation"] = "decisive_standard_base_outcome_memory_crossfit"
    summary["standard_pair_provenance"] = provenance
    summary["interpretation_constraints"].extend([
        "The memory was rebuilt exclusively from Standard Base versus Standard Alt outcomes.",
        "Historical conflict-audit and conflicting-rescue memories were rejected by manifest validation.",
        "The alternative prompt was frozen before this evaluation and is not the historical audit-pair action.",
    ])
    atomic_write_json(summary_path, summary)
    manifest_path = args.output_dir / "manifest.json"
    output_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_manifest["evaluation"] = summary["evaluation"]
    output_manifest["standard_pair_provenance"] = provenance
    atomic_write_json(manifest_path, output_manifest)
    print(f"STANDARD_PAIR_DONE rows={provenance['rows']} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
