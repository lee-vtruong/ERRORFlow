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


def aggregate_digest(rows):
    """Hash canonical manifest lines without depending on JSON formatting."""
    aggregate = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["path"]):
        aggregate.update(
            f'{row["path"]}\0{row["bytes"]}\0{row["sha256"]}\n'.encode("utf-8")
        )
    return aggregate.hexdigest()


def is_model_artifact(row):
    """Exclude mutable downloader state from the reproducible snapshot digest."""
    return not row["path"].startswith(".cache/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Recompute aggregate fields from an existing per-file manifest without rereading model bytes.",
    )
    args = parser.parse_args()
    root = args.model_dir.resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"No config.json under model snapshot: {root}")
    if args.reuse_existing:
        if not args.output.is_file():
            raise FileNotFoundError(f"Cannot reuse missing manifest: {args.output}")
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        rows = prior["files"]
        print(f"REUSED {len(rows)} per-file hashes from {args.output}", flush=True)
    else:
        files = sorted(path for path in root.rglob("*") if path.is_file())
        rows = []
        for index, path in enumerate(files, 1):
            relative = path.relative_to(root).as_posix()
            digest = digest_file(path)
            size = path.stat().st_size
            rows.append({"path": relative, "bytes": size, "sha256": digest})
            print(f"HASHED {index}/{len(files)} {relative}", flush=True)
    artifact_rows = [row for row in rows if is_model_artifact(row)]
    weight_rows = [row for row in artifact_rows if row["path"].endswith(".safetensors")]
    if not weight_rows:
        raise ValueError("No .safetensors weight files found in the snapshot manifest")
    report = {
        "schema_version": 2,
        "model_dir": str(root),
        "algorithm": "sha256_per_file_then_sha256_canonical_manifest_lines",
        "files": rows,
        "total_bytes": sum(row["bytes"] for row in rows),
        "model_artifact_bytes": sum(row["bytes"] for row in artifact_rows),
        "weight_bytes": sum(row["bytes"] for row in weight_rows),
        "snapshot_sha256": aggregate_digest(artifact_rows),
        "weights_sha256": aggregate_digest(weight_rows),
        "full_tree_sha256": aggregate_digest(rows),
        "weight_files": weight_rows,
        "excluded_from_snapshot": [row["path"] for row in rows if not is_model_artifact(row)],
        "reproducibility_note": (
            "snapshot_sha256 excludes mutable .cache downloader state; weights_sha256 "
            "covers only safetensors shards; full_tree_sha256 retains the complete audit tree"
        ),
    }
    atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "model_dir",
                    "model_artifact_bytes",
                    "weight_bytes",
                    "snapshot_sha256",
                    "weights_sha256",
                    "full_tree_sha256",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
