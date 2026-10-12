"""Small, safe command-line surface for V2 data and memory operations."""

from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path

from .adapters import get_adapter
from .error_memory import ErrorMemoryStore
from .io import config_hash, write_jsonl


def normalize(args: argparse.Namespace) -> None:
    adapter = get_adapter(args.dataset)
    examples = adapter.iter_file(args.input, args.split)
    if args.limit is not None:
        examples = islice(examples, args.limit)
    rows = (example.to_dict() for example in examples)
    write_jsonl(args.output, rows)
    print(f"NORMALIZE_DONE dataset={adapter.name} output={args.output}")


def inspect_memory(args: argparse.Namespace) -> None:
    store = ErrorMemoryStore.load(args.memory)
    neighbors = store.retrieve(args.claim, dataset=args.dataset, top_k=args.top_k)
    request = {"memory": str(args.memory), "claim": args.claim, "dataset": args.dataset, "top_k": args.top_k}
    print(json.dumps({"config_hash": config_hash(request), "neighbors": [
        {"score": score, "record": record.to_dict()} for record, score in neighbors
    ]}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="errorflow", description="ERRORFlow V2 utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    normalizer = commands.add_parser("normalize", help="Normalize a dataset JSONL file")
    normalizer.add_argument("--dataset", required=True, choices=["fever", "averitec", "scifact", "climate-fever"])
    normalizer.add_argument("--split", required=True)
    normalizer.add_argument("--input", required=True, type=Path)
    normalizer.add_argument("--output", required=True, type=Path)
    normalizer.add_argument("--limit", type=int, help="Normalize only the first N examples")
    normalizer.set_defaults(func=normalize)
    memory = commands.add_parser("inspect-memory", help="Inspect nearest compatible error memories")
    memory.add_argument("--memory", required=True, type=Path)
    memory.add_argument("--claim", required=True)
    memory.add_argument("--dataset")
    memory.add_argument("--top-k", type=int, default=5)
    memory.set_defaults(func=inspect_memory)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
