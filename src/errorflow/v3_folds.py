"""Deterministic group folds and contamination checks for V3."""

import hashlib
import re
from collections import defaultdict


def normalized_claim(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def claim_group(text: str) -> str:
    return hashlib.sha256(normalized_claim(text).encode("utf-8")).hexdigest()[:16]


def assign_grouped_folds(rows, n_folds=5, seed=2026):
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    groups = defaultdict(list)
    for row in rows:
        groups[claim_group(row["claim"])].append(str(row["example_id"]))
    folds = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest()))
    for _, ids in ordered:
        target = min(range(n_folds), key=lambda index: (sizes[index], index))
        folds[target].extend(sorted(ids))
        sizes[target] += len(ids)
    return {example_id: fold for fold, ids in enumerate(folds) for example_id in ids}


def validate_fold_isolation(memory_ids, evaluation_ids):
    overlap = set(memory_ids) & set(evaluation_ids)
    if overlap:
        raise ValueError(f"Fold contamination: {len(overlap)} evaluation IDs occur in memory")


def reject_opened_holdout(rows):
    forbidden = {f"dev:{index}" for index in range(25, 500)}
    overlap = forbidden & {str(row["example_id"]) for row in rows}
    if overlap:
        raise ValueError(f"V3 development input contains {len(overlap)} opened AVeriTeC holdout IDs")
