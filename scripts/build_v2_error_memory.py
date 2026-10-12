"""Build V2 error memory from paired experiment outcomes."""

import argparse

from errorflow.error_memory import ErrorMemoryStore
from errorflow.io import read_jsonl
from errorflow.memory_builder import build_memory_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--permit-evaluation-routing", action="store_true",
                        help="Unsafe for reported evaluation; diagnostics only.")
    args = parser.parse_args()
    records = build_memory_records(read_jsonl(args.input), dataset=args.dataset,
                                   source_split=args.source_split, action=args.action,
                                   permit_evaluation_routing=args.permit_evaluation_routing)
    ErrorMemoryStore(records).save(args.output)
    print(f"MEMORY_DONE records={len(records)} routable={sum(r.allow_for_routing for r in records)} output={args.output}")


if __name__ == "__main__":
    main()
