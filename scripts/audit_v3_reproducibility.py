"""Artifact-level leakage and metric audit for the frozen V3 AVeriTeC run."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl
from errorflow.v3_folds import assign_grouped_folds, normalized_claim
from errorflow.v3_memory import ResidualMemoryStore
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap, percentile
from scripts.materialize_v3_utility_policy import materialize, summarize


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def per_class(expected, predicted):
    report = {}
    for label in sorted(set(expected) | set(predicted)):
        tp = sum(g == p == label for g, p in zip(expected, predicted))
        fp = sum(g != label and p == label for g, p in zip(expected, predicted))
        fn = sum(g == label and p != label for g, p in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": sum(g == label for g in expected),
        }
    return report


def latency_summary(rows):
    values = sorted(float(row.get("latency_ms", 0.0)) for row in rows)
    return {
        "mean_ms": sum(values) / len(values) if values else 0.0,
        "p95_ms": percentile(values, 0.95) if values else 0.0,
        "measurement": "cached component latencies summed for routed examples; not fresh end-to-end wall clock",
    }


def duplicate_audit(rows, folds, threshold=0.9, example_limit=50):
    exact = defaultdict(list)
    tokens = {}
    for row in rows:
        key = str(row["example_id"])
        normalized = normalized_claim(row["claim"])
        exact[normalized].append(key)
        tokens[key] = set(normalized.split())
    exact_groups = [ids for ids in exact.values() if len(ids) > 1]
    cross_fold_exact = [ids for ids in exact_groups if len({folds[key] for key in ids}) > 1]
    keys = sorted(tokens)
    near_pairs = []
    for left_index, left in enumerate(keys):
        left_tokens = tokens[left]
        if not left_tokens:
            continue
        for right in keys[left_index + 1:]:
            if folds[left] == folds[right]:
                continue
            right_tokens = tokens[right]
            larger = max(len(left_tokens), len(right_tokens))
            smaller = min(len(left_tokens), len(right_tokens))
            if not right_tokens or smaller / larger < threshold:
                continue
            score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
            if score >= threshold:
                near_pairs.append({"left": left, "right": right, "token_jaccard": score})
    return {
        "exact_duplicate_groups": len(exact_groups),
        "cross_fold_exact_duplicate_groups": len(cross_fold_exact),
        "cross_fold_exact_examples": cross_fold_exact[:example_limit],
        "near_duplicate_threshold": threshold,
        "cross_fold_near_duplicate_pairs": len(near_pairs),
        "cross_fold_near_duplicate_examples": near_pairs[:example_limit],
        "near_duplicate_definition": "set-token Jaccard on normalized claims; descriptive lexical audit",
    }


def evidence_overlap_audit(gold_rows, folds, example_limit=50):
    fingerprints = defaultdict(lambda: {"folds": set(), "claims": set()})
    sources = defaultdict(lambda: {"folds": set(), "claims": set()})
    for row in gold_rows:
        key = str(row["example_id"])
        fold = folds[key]
        for evidence in row.get("evidence", []):
            if not isinstance(evidence, dict):
                text = str(evidence)
                evidence = {}
            else:
                text = str(evidence.get("text", ""))
            normalized = normalized_claim(text)
            if normalized:
                fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
                fingerprints[fingerprint]["folds"].add(fold)
                fingerprints[fingerprint]["claims"].add(key)
            source = evidence.get("url") or evidence.get("title")
            if source:
                sources[str(source)]["folds"].add(fold)
                sources[str(source)]["claims"].add(key)
    shared_evidence = [
        {"fingerprint": key, "folds": sorted(value["folds"]), "claim_count": len(value["claims"])}
        for key, value in fingerprints.items() if len(value["folds"]) > 1
    ]
    shared_sources = [
        {"source": key, "folds": sorted(value["folds"]), "claim_count": len(value["claims"])}
        for key, value in sources.items() if len(value["folds"]) > 1
    ]
    return {
        "evidence_fingerprints_shared_across_folds": len(shared_evidence),
        "shared_evidence_examples": sorted(shared_evidence, key=lambda row: (-row["claim_count"], row["fingerprint"]))[:example_limit],
        "source_identifiers_shared_across_folds": len(shared_sources),
        "shared_source_examples": sorted(shared_sources, key=lambda row: (-row["claim_count"], row["source"]))[:example_limit],
        "interpretation": "Shared evidence/source identifiers are reported for review and are not automatically treated as leakage.",
    }


def compare_rows(expected_rows, observed_rows):
    expected = indexed(expected_rows, "replayed policy")
    observed = indexed(observed_rows, "materialized policy")
    if set(expected) != set(observed):
        return {"id_match": False, "route_mismatches": None, "prediction_mismatches": None}
    return {
        "id_match": True,
        "route_mismatches": sum(bool(expected[key]["routed"]) != bool(observed[key].get("routed")) for key in expected),
        "prediction_mismatches": sum(expected[key]["prediction"] != observed[key]["prediction"] for key in expected),
    }


def audit(args):
    gold_rows = list(read_jsonl(args.gold))
    baseline_rows = list(read_jsonl(args.baseline_records))
    candidate_rows = list(read_jsonl(args.candidate))
    memory_rows = list(read_jsonl(args.memory))
    materialized_rows = list(read_jsonl(args.materialized))
    gold = indexed(gold_rows, "gold")
    baseline = indexed(baseline_rows, "baseline")
    candidate = indexed(candidate_rows, "candidate")
    materialized = indexed(materialized_rows, "materialized")
    ids = set(gold)
    deterministic_folds = assign_grouped_folds(gold_rows, args.folds, args.seed)
    recorded_folds = {key: baseline[key].get("outer_fold") for key in baseline}
    fold_assignment_mismatches = sum(recorded_folds.get(key) != deterministic_folds.get(key) for key in ids)

    memory_ids = [row["memory_id"] for row in memory_rows]
    memory_claim_ids = [str(row["source_claim_id"]) for row in memory_rows]
    memory_fold_mismatches = sum(
        row.get("provenance", {}).get("outer_fold") != deterministic_folds.get(str(row["source_claim_id"]))
        for row in memory_rows
    )
    fold_isolation = []
    for fold in range(args.folds):
        validation = {key for key, value in deterministic_folds.items() if value == fold}
        full_memory_sources = set(memory_claim_ids)
        usable_sources = {
            str(row["source_claim_id"]) for row in memory_rows
            if row.get("provenance", {}).get("outer_fold") != fold
        }
        fold_isolation.append({
            "outer_fold": fold,
            "validation_examples": len(validation),
            "full_memory_intersection": len(validation & full_memory_sources),
            "usable_memory_intersection_after_fold_exclusion": len(validation & usable_sources),
            "usable_memory_records": sum(row.get("provenance", {}).get("outer_fold") != fold for row in memory_rows),
        })

    store = ResidualMemoryStore.load(args.memory)
    replayed = materialize(
        baseline_rows, candidate_rows, store, args.action,
        args.top_k, args.min_support, args.cost_weight, args.utility_threshold,
        progress_every=0,
    )
    replay_comparison = compare_rows(replayed, materialized_rows)

    rotated = [dict(row) for row in baseline_rows]
    labels = [row["gold_label"] for row in rotated]
    for index, row in enumerate(rotated):
        row["gold_label"] = labels[(index + 1) % len(labels)]
    invariant_replay = materialize(
        rotated, candidate_rows, store, args.action,
        args.top_k, args.min_support, args.cost_weight, args.utility_threshold,
        progress_every=0,
    )
    gold_invariance = {
        "route_mismatches_after_rotating_target_gold": sum(
            left["routed"] != right["routed"] for left, right in zip(replayed, invariant_replay)
        ),
        "prediction_mismatches_after_rotating_target_gold": sum(
            left["prediction"] != right["prediction"] for left, right in zip(replayed, invariant_replay)
        ),
    }

    expected = [materialized[key]["gold_label"] for key in sorted(materialized)]
    before = [materialized[key]["baseline_prediction"] for key in sorted(materialized)]
    after = [materialized[key]["prediction"] for key in sorted(materialized)]
    recomputed_summary = summarize(materialized_rows)
    statistics = {
        "summary": recomputed_summary,
        "integer_identity": {
            "helpful_minus_harmful": recomputed_summary["helpful"] - recomputed_summary["harmful"],
            "net_over_n": recomputed_summary["net"] / len(materialized_rows),
            "reported_delta_accuracy": recomputed_summary["delta_accuracy"],
        },
        "per_class_baseline": per_class(expected, before),
        "per_class_candidate": per_class(expected, after),
        "mcnemar_exact": mcnemar_exact(expected, before, after),
        "paired_bootstrap": paired_bootstrap(expected, before, after, args.bootstrap_samples, args.seed),
        "latency": latency_summary(materialized_rows),
        "mean_tokens": sum(float(row.get("token_count", 0)) for row in materialized_rows) / len(materialized_rows),
    }
    reported_comparison = None
    if args.reported_significance:
        reported = json.loads(args.reported_significance.read_text(encoding="utf-8"))
        reported_comparison = {
            "path": str(args.reported_significance),
            "sha256": sha256_file(args.reported_significance),
            "accuracy_delta_abs_difference": abs(reported["delta_accuracy"] - statistics["summary"]["delta_accuracy"]),
            "macro_f1_delta_abs_difference": abs(reported["delta_macro_f1"] - statistics["summary"]["delta_macro_f1"]),
            "mcnemar_exact_match": reported["mcnemar_exact"] == statistics["mcnemar_exact"],
            "bootstrap_exact_match": reported["paired_bootstrap"] == statistics["paired_bootstrap"],
        }

    hard_checks = {
        "all_artifact_ids_match": set(baseline) == ids == set(candidate) == set(materialized),
        "fold_assignment_matches_seeded_algorithm": fold_assignment_mismatches == 0,
        "memory_ids_unique": len(memory_ids) == len(set(memory_ids)),
        "memory_claims_match_evaluation_population": set(memory_claim_ids) == ids,
        "memory_is_train_only": all(row.get("source_split") == "train" for row in memory_rows),
        "memory_contains_no_dev_ids": not any(key.startswith("dev:") for key in memory_claim_ids),
        "memory_fold_provenance_matches": memory_fold_mismatches == 0,
        "every_fold_has_zero_usable_id_overlap": all(
            row["usable_memory_intersection_after_fold_exclusion"] == 0 for row in fold_isolation
        ),
        "materialized_policy_replays_exactly": replay_comparison.get("route_mismatches") == 0 and replay_comparison.get("prediction_mismatches") == 0,
        "routing_is_invariant_to_target_gold": all(value == 0 for value in gold_invariance.values()),
    }
    hard_pass = all(hard_checks.values())
    return {
        "schema_version": 1,
        "audit": "ERRORFlow V3 AVeriTeC reproducibility and leakage audit",
        "git_head_supplied": args.git_head,
        "configuration": {
            "folds": args.folds, "seed": args.seed, "action": args.action,
            "top_k": args.top_k, "min_support": args.min_support,
            "cost_weight": args.cost_weight, "utility_threshold": args.utility_threshold,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in {
                "gold": args.gold, "baseline_records": args.baseline_records,
                "candidate": args.candidate, "memory": args.memory,
                "materialized": args.materialized,
            }.items()
        },
        "counts": {
            "gold": len(gold), "baseline": len(baseline), "candidate": len(candidate),
            "memory": len(memory_rows), "materialized": len(materialized),
            "labels": dict(Counter(row["label"] for row in gold_rows)),
        },
        "hard_checks": hard_checks,
        "hard_checks_pass": hard_pass,
        "fold_isolation": fold_isolation,
        "fold_assignment_mismatches": fold_assignment_mismatches,
        "memory_fold_mismatches": memory_fold_mismatches,
        "replay": replay_comparison,
        "target_gold_invariance": gold_invariance,
        "duplicates": duplicate_audit(gold_rows, deterministic_folds, args.near_duplicate_threshold),
        "evidence_overlap": evidence_overlap_audit(gold_rows, deterministic_folds),
        "statistics": statistics,
        "reported_significance_comparison": reported_comparison,
        "historical_result_classification": {
            "S0_and_B1_same_train_comparison": "DESCRIPTIVE_ONLY",
            "R6_fold_excluded_fixed_configuration": (
                "VERIFIED_BUT_TRAIN_SELECTED" if hard_pass else "INVALID_DUE_TO_LEAKAGE"
            ),
            "reason": (
                "Per-query outer-fold memory exclusion is audited, but A3 was selected on an AVeriTeC train screen "
                "and nested inner-fold model selection was not used."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--materialized", required=True, type=Path)
    parser.add_argument("--reported-significance", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--git-head", default="unknown")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--action", default="conflicting_rescue_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.9)
    args = parser.parse_args()
    report = audit(args)
    atomic_write_json(args.output, report)
    print(json.dumps({
        "hard_checks_pass": report["hard_checks_pass"],
        "classification": report["historical_result_classification"],
        "metrics": report["statistics"]["summary"],
        "duplicates": report["duplicates"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
