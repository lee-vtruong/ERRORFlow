"""Download and audit the small official CLIMATE-FEVER v1.0.1 artifact."""

import argparse
import hashlib
import json
import os
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY = "https://github.com/tdiggelm/climate-fever-dataset"
RELEASE = "1.0.1"
SOURCE = (
    "https://raw.githubusercontent.com/tdiggelm/climate-fever-dataset/"
    "1.0.1/dataset/climate-fever.jsonl"
)
LICENSE_STATUS = "unknown_no_license_declared_in_official_repository"
MAX_BYTES = 20 * 1024 * 1024


def parse_and_audit(data):
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    required = {"claim_id", "claim", "claim_label", "evidences"}
    for index, row in enumerate(rows):
        if not required <= row.keys():
            raise ValueError(f"Missing required fields at row {index}")
        if not isinstance(row["evidences"], list):
            raise ValueError(f"evidences must be a list at row {index}")
    ids = [str(row["claim_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Official CLIMATE-FEVER artifact contains duplicate claim IDs")
    labels = Counter(str(row["claim_label"]) for row in rows)
    evidence_labels = Counter(
        str(item.get("evidence_label")) for row in rows for item in row["evidences"]
    )
    evidence_counts = Counter(len(row["evidences"]) for row in rows)
    return rows, {
        "rows": len(rows), "unique_claim_ids": len(set(ids)),
        "claim_labels": dict(labels), "evidence_labels": dict(evidence_labels),
        "evidence_count_distribution": {str(key): value for key, value in sorted(evidence_counts.items())},
    }


def download(target, overwrite=False):
    if target.exists() and not overwrite:
        data, status = target.read_bytes(), "reused"
    else:
        request = urllib.request.Request(SOURCE, headers={"User-Agent": "ERRORFlow/0.3 dataset-audit"})
        with urllib.request.urlopen(request, timeout=120) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_BYTES:
                raise ValueError(f"Refusing file larger than {MAX_BYTES} bytes")
            data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError(f"Refusing file larger than {MAX_BYTES} bytes")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, target)
        status = "downloaded"
    _, audit = parse_and_audit(data)
    return {
        "url": SOURCE, "path": str(target), "status": status,
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), **audit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw/climate_fever")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_dir)
    artifact = download(root / "climate-fever.jsonl", args.overwrite)
    manifest = {
        "dataset": "CLIMATE-FEVER", "version": RELEASE,
        "source_repository": REPOSITORY, "license_status": LICENSE_STATUS,
        "redistribution_policy": "do_not_redistribute_dataset_from_ERRORFlow",
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "file": artifact,
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"CLIMATE_FEVER_READY manifest={manifest_path}")


if __name__ == "__main__":
    main()
