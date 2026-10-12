"""Small, safe command-line surface for V2 data and memory operations."""

from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import shutil
import sys

from .adapters import get_adapter
from .error_memory import ErrorMemoryStore
from .experiment import git_value
from .io import atomic_write_json, config_hash, write_jsonl
from .v3_prompts import B1_CONFLICT_AUDIT_SHA256, assert_b1_immutable, prompt_sha256, B1_CONFLICT_AUDIT


def normalize(args: argparse.Namespace) -> None:
    adapter = get_adapter(args.dataset)
    examples = adapter.iter_file(args.input, args.split)
    if args.offset < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError("--offset must be non-negative and --limit must be positive")
    stop = args.offset + args.limit if args.limit is not None else None
    examples = islice(examples, args.offset, stop)
    def validated_rows():
        seen = set()
        for example in examples:
            if not example.example_id or example.example_id == "None":
                raise ValueError("Adapter produced an empty example_id")
            if example.example_id in seen:
                raise ValueError(f"Adapter produced duplicate example_id: {example.example_id}")
            seen.add(example.example_id)
            yield example.to_dict()

    write_jsonl(args.output, validated_rows())
    print(f"NORMALIZE_DONE dataset={adapter.name} output={args.output}")


def inspect_memory(args: argparse.Namespace) -> None:
    store = ErrorMemoryStore.load(args.memory)
    neighbors = store.retrieve(args.claim, dataset=args.dataset, top_k=args.top_k)
    request = {"memory": str(args.memory), "claim": args.claim, "dataset": args.dataset, "top_k": args.top_k}
    print(json.dumps({"config_hash": config_hash(request), "neighbors": [
        {"score": score, "record": record.to_dict()} for record, score in neighbors
    ]}, ensure_ascii=False, indent=2))


def doctor(args: argparse.Namespace) -> None:
    assert_b1_immutable()
    usage = shutil.disk_usage(Path.cwd())
    report = {
        "status": "ok", "python": sys.version.split()[0], "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "disk_free_gb": round(usage.free / 1024 ** 3, 2),
        "b1_prompt_sha256": prompt_sha256(B1_CONFLICT_AUDIT),
        "b1_prompt_immutable": prompt_sha256(B1_CONFLICT_AUDIT) == B1_CONFLICT_AUDIT_SHA256,
    }
    print(json.dumps(report, indent=2))


def v3_audit(args: argparse.Namespace) -> None:
    assert_b1_immutable()
    tag = "archive/errorflow-v2-conflict-audit"
    report = {
        "phase": "V3_phase_0_repository_audit", "status": "completed",
        "head": git_value("rev-parse", "HEAD"), "git_dirty": bool(git_value("status", "--porcelain")),
        "v2_archive_tag": tag, "v2_archive_commit": git_value("rev-list", "-n", "1", tag),
        "b1_prompt_sha256": B1_CONFLICT_AUDIT_SHA256, "b1_prompt_bytes": len(B1_CONFLICT_AUDIT.encode("utf-8")),
        "opened_holdout_policy": "dev:25..dev:499 excluded from V3 development",
        "historical_reports": [
            "docs/reports/AVERITEC_V2_HOLDOUT_REPORT.md", "docs/STEP_48_V2_HOLDOUT_ONE_SHOT.md",
            "docs/reports/V2_IMPLEMENTATION_LOG.md",
        ],
    }
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="errorflow", description="ERRORFlow reproducible research utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    health = commands.add_parser("doctor", help="Audit local runtime and frozen prompt invariants")
    health.set_defaults(func=doctor)
    normalizer = commands.add_parser("normalize", help="Normalize a dataset JSONL file")
    normalizer.add_argument("--dataset", required=True, choices=["fever", "averitec", "scifact", "climate-fever"])
    normalizer.add_argument("--split", required=True)
    normalizer.add_argument("--input", required=True, type=Path)
    normalizer.add_argument("--output", required=True, type=Path)
    normalizer.add_argument("--limit", type=int, help="Normalize only the first N examples")
    normalizer.add_argument("--offset", type=int, default=0, help="Skip the first N examples")
    normalizer.set_defaults(func=normalize)
    memory = commands.add_parser("inspect-memory", help="Inspect nearest compatible error memories")
    memory.add_argument("--memory", required=True, type=Path)
    memory.add_argument("--claim", required=True)
    memory.add_argument("--dataset")
    memory.add_argument("--top-k", type=int, default=5)
    memory.set_defaults(func=inspect_memory)
    v3 = commands.add_parser("v3", help="ERRORFlow V3 research commands")
    v3_commands = v3.add_subparsers(dest="v3_command", required=True)
    audit = v3_commands.add_parser("audit", help="Verify the V2 boundary and V3 contamination policy")
    audit.add_argument("--output", type=Path)
    audit.set_defaults(func=v3_audit)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
