"""Fail on missing citation keys, duplicate BibTeX keys, or uncited entries."""
from __future__ import annotations
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
tex = "\n".join(p.read_text(encoding="utf-8") for p in root.rglob("*.tex"))
bib = (root / "references.bib").read_text(encoding="utf-8")
keys = re.findall(r"@[A-Za-z]+\s*\{\s*([^,]+),", bib)
cited = set()
for group in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex):
    cited.update(k.strip() for k in group.split(","))
duplicates = sorted({k for k in keys if keys.count(k) > 1})
missing = sorted(cited - set(keys))
uncited = sorted(set(keys) - cited)
print(f"bib_entries={len(keys)} cited={len(cited)} duplicates={duplicates} missing={missing} uncited={uncited}")
raise SystemExit(bool(duplicates or missing or uncited))
