# FEVER V0/V1 legacy archive

This directory freezes the reproducibility boundary that existed immediately
before the ERRORFlow V2 migration on 2026-10-12.

The archive is intentionally reference-based. Source code is pinned by commit
and SHA-256 checksums; large datasets, indexes, models, and experiment outputs
remain in their original server locations. This avoids duplicating roughly
14 GB of FEVER data/indexes and the model cache on a server with limited free
space.

## Frozen boundary

- Repository: `https://github.com/lee-vtruong/ERRORFlow`
- Branch: `main`
- Audited commit: `a4280b8ac550909beb57b5509e10bf71cd289304`
- Local test result: `32 passed, 11 warnings`
- Test date: 2026-10-12 (Asia/Saigon)
- Git tag: not created during the audit because the working tree contains
  user-owned changes and `.git` is read-only in the current workspace.

## Contents

- `manifests/files_manifest.json`: source and external artifact inventory.
- `manifests/historical_results.json`: verified FEVER result ledger.
- `manifests/checksums.sha256`: practical integrity anchors.
- `manifests/environment_snapshot.txt`: local and server environments.
- `configs/`: frozen human-readable V0/V1 configurations.
- `scripts/README.md`: mapping from stages to the original scripts.
- `original_commands.md`: commands used for the historical runs.
- `reports/FEVER_V1_ARCHIVE_REPORT.md`: archive verification and rollback.

No file in this archive implies that a server-only artifact was copied or
re-hashed locally. Availability and verification level are explicit in the
manifest.

Verify source anchors against the frozen commit (not the current V2 working
tree) with:

```bash
python scripts/verify_fever_v1_archive.py
```
