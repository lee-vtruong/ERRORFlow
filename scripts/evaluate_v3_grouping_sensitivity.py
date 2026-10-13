"""Re-evaluate fixed V3 R6 under near-duplicate-grouped outer folds."""

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_folds import normalized_claim
from errorflow.v3_memory import ResidualMemoryStore, record_from_dict
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap

try:
    from scripts.materialize_v3_utility_policy import materialize, summarize
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from materialize_v3_utility_policy import materialize, summarize


class DisjointSet:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root))
            self.parent[larger] = smaller


def near_duplicate_edges(rows, minimum_threshold):
    tokens = {
        str(row["example_id"]): set(normalized_claim(row["claim"]).split())
        for row in rows
    }
    keys = sorted(tokens)
    edges = []
    for left_index, left in enumerate(keys):
        left_tokens = tokens[left]
        if not left_tokens:
            continue
        for right in keys[left_index + 1:]:
            right_tokens = tokens[right]
            if not right_tokens:
                continue
            larger = max(len(left_tokens), len(right_tokens))
            smaller = min(len(left_tokens), len(right_tokens))
            if smaller / larger < minimum_threshold:
                continue
            score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
            if score >= minimum_threshold:
                edges.append((left, right, score))
    return edges


def grouped_folds(rows, edges, threshold, n_folds=5, seed=2026):
    keys = [str(row["example_id"]) for row in rows]
    groups = DisjointSet(keys)
    for left, right, score in edges:
        if score >= threshold:
            groups.union(left, right)
    components = defaultdict(list)
    for key in keys:
        components[groups.find(key)].append(key)
    folds = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    ordered = sorted(
        components.values(),
        key=lambda members: (
            -len(members),
            hashlib.sha256(f"{seed}:{'|'.join(sorted(members))}".encode()).hexdigest(),
        ),
    )
    for members in ordered:
        target = min(range(n_folds), key=lambda index: (sizes[index], index))
        folds[target].extend(members)
        sizes[target] += len(members)
    assignments = {key: fold for fold, members in enumerate(folds) for key in members}
    return assignments, [sorted(members) for members in components.values()]


def threshold_slug(value):
    return f"{value:.2f}".replace(".", "p")


def evaluate_threshold(
    baseline_rows, candidate_rows, memory_rows, original_rows, edges,
    threshold, folds, seed, action, top_k, min_support, cost_weight,
    utility_threshold, bootstrap_samples,
):
    assignments, components = grouped_folds(baseline_rows, edges, threshold, folds, seed)
    updated_baseline = []
    previous_folds = {}
    for source in baseline_rows:
        row = dict(source)
        key = str(row["example_id"])
        previous_folds[key] = row.get("outer_fold")
        row["outer_fold"] = assignments[key]
        updated_baseline.append(row)
    updated_memory = []
    for payload in memory_rows:
        record = record_from_dict(payload)
        provenance = dict(record.provenance)
        provenance["outer_fold"] = assignments[record.source_claim_id]
        provenance["fold_grouping_sensitivity"] = {
            "method": "normalized_claim_token_jaccard_connected_components",
            "threshold": threshold,
            "seed": seed,
        }
        updated_memory.append(replace(record, provenance=provenance))
    rows = materialize(
        updated_baseline, candidate_rows, ResidualMemoryStore(updated_memory), action,
        top_k, min_support, cost_weight, utility_threshold, progress_every=0,
    )
    metrics = summarize(rows)
    ordered = sorted(rows, key=lambda row: str(row["example_id"]))
    expected = [row["gold_label"] for row in ordered]
    before = [row["baseline_prediction"] for row in ordered]
    after = [row["prediction"] for row in ordered]
    original = {str(row["example_id"]): row for row in original_rows}
    route_overlap = None
    if original:
        selected = {str(row["example_id"]) for row in rows if row["routed"]}
        old_selected = {key for key, row in original.items() if row.get("routed")}
        route_overlap = {
            "original_routed": len(old_selected),
            "sensitivity_routed": len(selected),
            "intersection": len(old_selected & selected),
            "jaccard": len(old_selected & selected) / len(old_selected | selected) if old_selected | selected else 1.0,
        }
    isolation = []
    for fold in range(folds):
        validation = {key for key, value in assignments.items() if value == fold}
        usable = {
            record.source_claim_id for record in updated_memory
            if record.provenance.get("outer_fold") != fold
        }
        isolation.append({
            "fold": fold,
            "validation_examples": len(validation),
            "usable_validation_id_overlap": len(validation & usable),
        })
    grouped_components = [members for members in components if len(members) > 1]
    return rows, {
        "threshold": threshold,
        "grouping": "normalized_claim_token_jaccard_connected_components",
        "components": len(components),
        "multi_example_components": len(grouped_components),
        "largest_component": max(map(len, components)),
        "examples_in_multi_example_components": sum(map(len, grouped_components)),
        "fold_sizes": dict(sorted({fold: sum(value == fold for value in assignments.values()) for fold in range(folds)}.items())),
        "examples_with_changed_fold_id": sum(previous_folds[key] != assignments[key] for key in assignments),
        "isolation": isolation,
        "metrics": metrics,
        "mcnemar_exact": mcnemar_exact(expected, before, after),
        "paired_bootstrap": paired_bootstrap(expected, before, after, bootstrap_samples, seed),
        "route_overlap_with_historical": route_overlap,
        "interpretation": "post-audit grouping sensitivity; not a newly selected primary estimate",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--original-materialized", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.85, 0.9, 0.95])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--action", default="conflicting_rescue_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    if not args.thresholds or any(not 0.0 <= value <= 1.0 for value in args.thresholds):
        raise ValueError("thresholds must be within [0, 1]")
    baseline_rows = list(read_jsonl(args.baseline_records))
    candidate_rows = list(read_jsonl(args.candidate))
    memory_rows = list(read_jsonl(args.memory))
    original_rows = list(read_jsonl(args.original_materialized))
    edges = near_duplicate_edges(baseline_rows, min(args.thresholds))
    reports = []
    for threshold in sorted(set(args.thresholds)):
        rows, report = evaluate_threshold(
            baseline_rows, candidate_rows, memory_rows, original_rows, edges,
            threshold, args.folds, args.seed, args.action, args.top_k,
            args.min_support, args.cost_weight, args.utility_threshold,
            args.bootstrap_samples,
        )
        slug = threshold_slug(threshold)
        write_jsonl(args.output_dir / f"jaccard_{slug}_predictions.jsonl", rows)
        atomic_write_json(args.output_dir / f"jaccard_{slug}_metrics.json", report)
        reports.append(report)
    summary = {
        "evaluation": "post_audit_near_duplicate_grouping_sensitivity",
        "edge_search_minimum_threshold": min(args.thresholds),
        "candidate_edges": len(edges),
        "results": reports,
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
