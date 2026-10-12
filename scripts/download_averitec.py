"""Download only the small official AVeriTeC claim files and audit them."""

import argparse
import hashlib
import json
import os
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = "https://github.com/MichSchli/AVeriTeC"
LICENSE = "CC BY-NC 4.0"
SOURCES = {
    "train": "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/train.json",
    "dev": "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json",
}
MAX_BYTES = 100 * 1024 * 1024


def download(url: str, target: Path, overwrite: bool = False) -> dict:
    if target.exists() and not overwrite:
        data, status = target.read_bytes(), "reused"
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "ERRORFlow/0.2 dataset-audit"})
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
    rows = json.loads(data)
    if not isinstance(rows, list):
        raise ValueError(f"Expected JSON array from {url}")
    invalid = [i for i, row in enumerate(rows) if not {"claim", "label"} <= row.keys()]
    if invalid:
        raise ValueError(f"Missing required fields in {target}; first invalid index={invalid[0]}")
    return {"url": url, "path": str(target), "status": status, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows),
            "labels": dict(Counter(str(row["label"]) for row in rows))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw/averitec")
    parser.add_argument("--splits", nargs="+", choices=sorted(SOURCES), default=["train", "dev"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_dir)
    files = [download(SOURCES[split], root / f"{split}.json", args.overwrite) for split in args.splits]
    manifest = {"dataset": "AVeriTeC", "source_repository": REPOSITORY, "license": LICENSE,
                "source_revision": "main_with_content_sha256", "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "files": files}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"AVERITEC_READY manifest={path}")


if __name__ == "__main__":
    main()
