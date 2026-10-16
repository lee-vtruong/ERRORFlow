"""Create byte-level SHA-256 checksums for a frozen local model snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from errorflow.io import atomic_write_json


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.model_dir.resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"No config.json under model snapshot: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows, aggregate = [], hashlib.sha256()
    for index, path in enumerate(files, 1):
        relative = path.relative_to(root).as_posix()
        digest = digest_file(path)
        size = path.stat().st_size
        rows.append({"path": relative, "bytes": size, "sha256": digest})
        aggregate.update(f"{relative}\0{size}\0{digest}\n".encode("utf-8"))
        print(f"HASHED {index}/{len(files)} {relative}", flush=True)
    report = {
        "schema_version": 1,
        "model_dir": str(root),
        "algorithm": "sha256_per_file_then_sha256_canonical_manifest_lines",
        "files": rows,
        "total_bytes": sum(row["bytes"] for row in rows),
        "snapshot_sha256": aggregate.hexdigest(),
    }
    atomic_write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("model_dir", "total_bytes", "snapshot_sha256")}, indent=2))


if __name__ == "__main__":
    main()
