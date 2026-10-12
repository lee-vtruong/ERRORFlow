"""Verify legacy checksum anchors against the archived Git commit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "legacy" / "fever_v1"


def main() -> None:
    manifest = json.loads((ARCHIVE / "manifests" / "files_manifest.json").read_text(encoding="utf-8"))
    commit = manifest["audited_commit"]
    failures = []
    checked = 0
    for line in (ARCHIVE / "manifests" / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected, path = line.split(None, 1)
        path = path.strip()
        try:
            content = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
        except subprocess.CalledProcessError:
            failures.append({"path": path, "reason": "missing_from_commit"})
            continue
        actual = hashlib.sha256(content).hexdigest()
        checked += 1
        if actual != expected:
            failures.append({"path": path, "expected": expected, "actual": actual})
    if failures:
        raise SystemExit(f"ARCHIVE_INVALID checked={checked} failures={json.dumps(failures)}")
    print(f"ARCHIVE_OK commit={commit} checked={checked}")


if __name__ == "__main__":
    main()
