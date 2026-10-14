"""Precompute frozen claim/evidence embeddings for Group-C ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl


def indexed(rows):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id: {key}")
        result[key] = row
    return result


def evidence_text(row):
    return "\n".join(
        item.get("text", "") if isinstance(item, dict) else str(item)
        for item in row.get("evidence", [])
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True,
                        help="Pinned Hub revision/commit, or 'local:<sha256-or-release-id>' for a local model")
    parser.add_argument("--embedding-kind", required=True, choices=["sentence_bert", "e5"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.model_revision.strip():
        raise ValueError("--model-revision must be non-empty so the encoder is frozen audibly")
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("Install the project v2 extras: pip install -e '.[v2]'") from error

    baseline = indexed(read_jsonl(args.baseline_records))
    examples = indexed(read_jsonl(args.examples))
    if set(baseline) != set(examples):
        raise ValueError("Baseline and examples IDs must match exactly")
    ids = sorted(baseline)
    claims = [baseline[key]["claim"] for key in ids]
    evidence = [evidence_text(examples[key]) for key in ids]
    if args.embedding_kind == "e5":
        claims = ["query: " + text for text in claims]
        evidence = ["passage: " + text for text in evidence]
    revision = None if args.model_revision.startswith("local:") else args.model_revision
    model = SentenceTransformer(args.model, revision=revision, device=args.device)
    encode = dict(batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True)
    claim_vectors = model.encode(claims, **encode)
    evidence_vectors = model.encode(evidence, **encode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=np.asarray(ids),
        claim=np.asarray(claim_vectors, dtype=np.float32),
        evidence=np.asarray(evidence_vectors, dtype=np.float32),
    )
    actual_output = args.output if str(args.output).endswith(".npz") else Path(str(args.output) + ".npz")
    manifest = {
        "schema_version": 1,
        "embedding_kind": args.embedding_kind,
        "model": args.model,
        "model_revision": args.model_revision,
        "rows": len(ids),
        "dimensions": int(claim_vectors.shape[1]),
        "normalized": True,
        "e5_prefixes": args.embedding_kind == "e5",
        "baseline_records": str(args.baseline_records),
        "baseline_sha256": sha256(args.baseline_records),
        "examples": str(args.examples),
        "examples_sha256": sha256(args.examples),
        "output": str(actual_output),
        "output_sha256": sha256(actual_output),
    }
    atomic_write_json(Path(str(actual_output) + ".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
