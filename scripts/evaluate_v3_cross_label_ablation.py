"""Evaluate mechanisms enabled by the final no-label-filter memory router.

The evaluator replays cached B1 and A3 predictions.  It never invokes the
language model and never reads external evaluation labels for routing.  Every
memory lookup excludes the query claim and its grouped outer fold.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_memory import ResidualMemoryStore
from errorflow.v3_routing import estimate_action
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap

try:
    from scripts.evaluate_v3_component_ablation import route_overlap, top_count
    from scripts.evaluate_v3_matched_budget import (
        evaluate_selection,
        holm_adjust,
        indexed,
        prepare_grouped_artifacts,
    )
    from scripts.evaluate_v3_memory_ablation import FULL_WEIGHTS, shuffle_outcomes
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    from evaluate_v3_component_ablation import route_overlap, top_count
    from evaluate_v3_matched_budget import (
        evaluate_selection,
        holm_adjust,
        indexed,
        prepare_grouped_artifacts,
    )
    from evaluate_v3_memory_ablation import FULL_WEIGHTS, shuffle_outcomes


PRIMARY_VARIANTS = (
    "NB0_full_nlf",
    "NB1_hard_label",
    "NB2_cross_label_only",
    "NB3_cross_label_suppression",
    "NB4_cross_label_outcome_shuffle",
    "NB5_same_label_priority",
    "NB6_soft_label_penalty",
    "NB7_cross_label_quota1",
    "NB7_cross_label_quota2",
)


def query_for(key, row):
    return {
        "example_id": key,
        "claim": row["claim"],
        "retrieval_features": {
            "baseline_prediction": row["prediction"],
            "label_scores": row.get("label_scores", {}),
            "evidence_tokens": row.get("evidence_tokens", 0),
        },
    }


def same_label(record, query_label):
    return record.retrieval_features.get("baseline_prediction") == query_label


def estimate_bundle(neighbors, action, query_label, min_support, cost_weight):
    """Return utility plus an additive outcome-mass decomposition.

    The same/cross contributions decompose the smoothed net-outcome numerator
    over the common denominator ``W + 2``.  The symmetric pseudo-counts cancel
    in recovery minus regression.  The constant cost penalty is reported
    separately rather than arbitrarily assigned to either label group.
    """
    rows = [(record, float(weight)) for record, weight in neighbors
            if record.correction_strategy == action]
    estimate = estimate_action(action, rows, cost_weight)
    effective = rows
    weight = sum(value for _, value in rows)
    if rows and weight <= 0.0:
        effective = [(record, 1.0) for record, _ in rows]
        weight = float(len(rows))
    denominator = weight + 2.0

    def contribution(want_same):
        return sum(
            value * (record.recovery_count - record.regression_count)
            for record, value in effective
            if same_label(record, query_label) is want_same
        ) / denominator if effective else 0.0

    score = estimate.utility if estimate.support >= min_support else float("-inf")
    cross_mass = sum(
        value for record, value in rows if not same_label(record, query_label)
    )
    same_mass = sum(
        value for record, value in rows if same_label(record, query_label)
    )
    return {
        "score": score,
        "support": estimate.support if estimate.support >= min_support else 0,
        "uniform_weight_fallback": estimate.uniform_weight_fallback,
        "same_label_utility_contribution": contribution(True),
        "cross_label_utility_contribution": contribution(False),
        "estimated_cost_penalty": cost_weight * estimate.expected_cost,
        "same_label_similarity_mass": same_mass,
        "cross_label_similarity_mass": cross_mass,
        "cross_label_count_at_5": sum(
            not same_label(record, query_label) for record, _ in rows
        ),
        "neighbors": rows,
    }


def merge_with_penalty(same, cross, top_k, penalty):
    rows = list(same) + [(record, score * penalty) for record, score in cross]
    return sorted(rows, key=lambda item: (-item[1], item[0].memory_id))[:top_k]


def quota_neighbors(same, cross, top_k, quota):
    rows = sorted(list(same) + list(cross), key=lambda item: (-item[1], item[0].memory_id))
    output, used_cross = [], 0
    for record, score in rows:
        is_cross = not any(record.memory_id == candidate.memory_id for candidate, _ in same)
        if is_cross and used_cross >= quota:
            continue
        output.append((record, score))
        used_cross += int(is_cross)
        if len(output) == top_k:
            break
    return output


def route_outcome(key, selected, baseline, candidate):
    if key not in selected:
        return "not_routed"
    gold = baseline[key]["gold_label"]
    before = baseline[key]["prediction"]
    after = candidate[key]["prediction"]
    if before != gold and after == gold:
        return "helpful"
    if before == gold and after != gold:
        return "harmful"
    return "neutral"


def finite_or_none(value):
    return value if math.isfinite(value) else None


def summarize_replicates(rows):
    output = {"seeds": len(rows)}
    for field in ("routed", "route_rate", "helpful", "harmful", "net", "accuracy", "macro_f1"):
        values = [float(row[field]) for row in rows]
        output[field] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return output


def format_results(report):
    lines = [
        "ERRORFlow V3 N-B cross-label memory mechanism",
        f"matched_route_count={report['configuration']['matched_route_count']}",
        "",
        "VARIANT\tMODE\tROUTED\tRATE\tHELPFUL\tHARMFUL\tNET\tACC\tMACRO_F1\tROUTING_MS",
    ]
    for variant in PRIMARY_VARIANTS:
        seed_suffix = (
            f":seed={report['configuration']['seed']}"
            if variant.startswith("NB4_") else ""
        )
        for mode in ("natural", "matched_budget"):
            row = report["primary_results"][f"{variant}:{mode}{seed_suffix}"]
            lines.append("\t".join(map(str, (
                variant, mode, row["routed"], f"{row['route_rate']:.6f}",
                row["helpful"], row["harmful"], row["net"],
                f"{row['accuracy']:.6f}", f"{row['macro_f1']:.6f}",
                f"{row['mean_routing_latency_ms']:.4f}",
            ))))
    lines.extend(("", "MATCHED SIGNIFICANCE: NB0 MINUS CONTROL"))
    for variant, row in report["matched_significance_nb0_minus_variant"].items():
        boot = row["paired_bootstrap"]
        lines.append(
            f"{variant}\tholm_acc_p={row['holm_adjusted_mcnemar_p']}"
            f"\tacc_ci={boot['accuracy_delta_ci95']}"
            f"\tf1_ci={boot['macro_f1_delta_ci95']}"
        )
    lines.extend(("", "MECHANISM SUMMARY", json.dumps(
        report["mechanism_summary"], sort_keys=True
    )))
    lines.extend(("", "NB4 MULTI-SEED", json.dumps(
        report["nb4_stochastic_aggregate"], sort_keys=True
    )))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--action", default="conflicting_rescue_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--soft-label-penalty", type=float, default=0.5)
    parser.add_argument("--group-threshold", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--shuffle-seeds", nargs="+", type=int, default=list(range(2026, 2046)))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    if not 0.0 <= args.soft_label_penalty <= 1.0:
        raise ValueError("soft-label-penalty must be in [0, 1]")
    if args.top_k < args.min_support:
        raise ValueError("top-k must be at least min-support")

    baseline_rows = list(read_jsonl(args.baseline_records))
    candidate_rows = list(read_jsonl(args.candidate))
    memory_rows = list(read_jsonl(args.memory))
    baseline_rows, records, _, grouping = prepare_grouped_artifacts(
        baseline_rows, memory_rows, args.group_threshold, args.folds, args.seed,
    )
    baseline = indexed(baseline_rows, "baseline")
    candidate = indexed(candidate_rows, "candidate")
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate IDs must match exactly")
    ids = sorted(baseline)
    gold = [baseline[key]["gold_label"] for key in ids]
    base_metrics, _ = evaluate_selection(ids, baseline, candidate, set())
    store = ResidualMemoryStore(records)
    shuffled_maps = {
        seed: {record.memory_id: record for record in shuffle_outcomes(records, seed)}
        for seed in args.shuffle_seeds
    }

    bundles = {variant: {} for variant in PRIMARY_VARIANTS if not variant.startswith("NB4_")}
    shuffle_bundles = {seed: {} for seed in args.shuffle_seeds}
    mechanism = {}
    started = time.perf_counter()
    for index, key in enumerate(ids, 1):
        row = baseline[key]
        query = query_for(key, row)
        common = {
            "top_k": args.top_k,
            "action": args.action,
            "exclude_claim_ids": {key},
            "exclude_folds": {row["outer_fold"]},
            "similarity_weights": FULL_WEIGHTS,
        }
        lookup_started = time.perf_counter()
        nlf = store.retrieve(query, compatibility_mode="none", **common)
        nlf_latency_ms = (time.perf_counter() - lookup_started) * 1000.0
        lookup_started = time.perf_counter()
        same = store.retrieve(query, compatibility_mode="same_label", **common)
        same_latency_ms = (time.perf_counter() - lookup_started) * 1000.0
        lookup_started = time.perf_counter()
        cross = store.retrieve(query, compatibility_mode="different_label", **common)
        cross_latency_ms = (time.perf_counter() - lookup_started) * 1000.0
        priority = list(same[:args.top_k])
        if len(priority) < args.top_k:
            priority.extend(cross[:args.top_k - len(priority)])
        variants = {
            "NB0_full_nlf": nlf,
            "NB1_hard_label": same,
            "NB2_cross_label_only": cross,
            # No backfill: support is evaluated only on same-label members of
            # the already frozen NLF top-k neighborhood.
            "NB3_cross_label_suppression": [
                item for item in nlf if same_label(item[0], row["prediction"])
            ],
            "NB5_same_label_priority": priority,
            "NB6_soft_label_penalty": merge_with_penalty(
                same, cross, args.top_k, args.soft_label_penalty
            ),
            "NB7_cross_label_quota1": quota_neighbors(same, cross, args.top_k, 1),
            "NB7_cross_label_quota2": quota_neighbors(same, cross, args.top_k, 2),
        }
        for variant, neighbors in variants.items():
            bundles[variant][key] = estimate_bundle(
                neighbors, args.action, row["prediction"], args.min_support, args.cost_weight
            )
        variant_latencies = {
            "NB0_full_nlf": nlf_latency_ms,
            "NB1_hard_label": same_latency_ms,
            "NB2_cross_label_only": cross_latency_ms,
            "NB3_cross_label_suppression": nlf_latency_ms,
            "NB5_same_label_priority": (
                same_latency_ms if len(same) >= args.top_k
                else same_latency_ms + cross_latency_ms
            ),
            "NB6_soft_label_penalty": same_latency_ms + cross_latency_ms,
            "NB7_cross_label_quota1": same_latency_ms + cross_latency_ms,
            "NB7_cross_label_quota2": same_latency_ms + cross_latency_ms,
        }
        for variant, latency in variant_latencies.items():
            bundles[variant][key]["routing_latency_ms"] = latency
        for seed, shuffled in shuffled_maps.items():
            mixed = [
                (record if same_label(record, row["prediction"]) else shuffled[record.memory_id], score)
                for record, score in nlf
            ]
            shuffle_bundles[seed][key] = estimate_bundle(
                mixed, args.action, row["prediction"], args.min_support, args.cost_weight
            )
            shuffle_bundles[seed][key]["routing_latency_ms"] = nlf_latency_ms
        nb0 = bundles["NB0_full_nlf"][key]
        total_mass = nb0["same_label_similarity_mass"] + nb0["cross_label_similarity_mass"]
        mechanism[key] = {
            "example_id": key,
            "baseline_prediction": row["prediction"],
            "outer_fold": row["outer_fold"],
            "cross_label_count@5": nb0["cross_label_count_at_5"],
            "cross_label_count_at_5": nb0["cross_label_count_at_5"],
            "cross_label_similarity_mass": nb0["cross_label_similarity_mass"],
            "cross_label_similarity_fraction": (
                nb0["cross_label_similarity_mass"] / total_mass if total_mass else 0.0
            ),
            "same_label_utility_contribution": nb0["same_label_utility_contribution"],
            "cross_label_utility_contribution": nb0["cross_label_utility_contribution"],
            "estimated_cost_penalty": nb0["estimated_cost_penalty"],
            "nb0_neighbor_ids": [record.memory_id for record, _ in nb0["neighbors"]],
            "nb0_neighbor_labels": [
                record.retrieval_features.get("baseline_prediction")
                for record, _ in nb0["neighbors"]
            ],
            "nb0_neighbor_similarities": [score for _, score in nb0["neighbors"]],
        }
        if index % 250 == 0 or index == len(ids):
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0.0
            eta = (len(ids) - index) / rate if rate else 0.0
            print(
                f"NB_COMPUTE progress={index}/{len(ids)} elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
                flush=True,
            )

    primary_seed_bundles = shuffle_bundles[args.seed]
    all_bundles = dict(bundles)
    all_bundles["NB4_cross_label_outcome_shuffle"] = primary_seed_bundles
    nb0_scores = {key: bundle["score"] for key, bundle in bundles["NB0_full_nlf"].items()}
    nb0_natural = {key for key in ids if nb0_scores[key] > args.utility_threshold}
    matched_count = len(nb0_natural)
    results, routes, prediction_rows, prediction_index = {}, [], [], {}

    def record_run(variant, mode, seed, selected, variant_bundles, save_predictions=True):
        metrics, predictions = evaluate_selection(ids, baseline, candidate, selected)
        metrics["delta_accuracy_vs_b1"] = metrics["accuracy"] - base_metrics["accuracy"]
        metrics["delta_macro_f1_vs_b1"] = metrics["macro_f1"] - base_metrics["macro_f1"]
        metrics["route_overlap_with_nb0_natural"] = route_overlap(selected, nb0_natural)
        metrics["seed"] = seed
        latencies = [variant_bundles[key].get("routing_latency_ms", 0.0) for key in ids]
        metrics["mean_routing_latency_ms"] = statistics.mean(latencies)
        name = f"{variant}:{mode}" + (f":seed={seed}" if seed is not None else "")
        results[name] = metrics
        prediction_index[(variant, mode, seed)] = predictions
        routes.append({
            "variant": variant, "mode": mode, "seed": seed,
            "selected_ids": sorted(selected),
        })
        if save_predictions:
            for key, prediction in zip(ids, predictions):
                bundle = variant_bundles[key]
                prediction_rows.append({
                    "example_id": key,
                    "variant": variant,
                    "mode": mode,
                    "seed": seed,
                    "routed": key in selected,
                    "prediction": prediction,
                    "gold_label": baseline[key]["gold_label"],
                    "estimated_utility": finite_or_none(bundle["score"]),
                    "estimated_support": bundle["support"],
                    "cross_label_count@5": bundle["cross_label_count_at_5"],
                    "cross_label_count_at_5": bundle["cross_label_count_at_5"],
                    "cross_label_similarity_mass": bundle["cross_label_similarity_mass"],
                    "same_label_utility_contribution": bundle["same_label_utility_contribution"],
                    "cross_label_utility_contribution": bundle["cross_label_utility_contribution"],
                    "route_outcome": route_outcome(key, selected, baseline, candidate),
                })
        return metrics

    for variant, variant_bundles in all_bundles.items():
        scores = {key: bundle["score"] for key, bundle in variant_bundles.items()}
        natural = {key for key in ids if scores[key] > args.utility_threshold}
        matched = top_count(ids, scores, matched_count)
        seed = args.seed if variant.startswith("NB4_") else None
        record_run(variant, "natural", seed, natural, variant_bundles)
        record_run(variant, "matched_budget", seed, matched, variant_bundles)

    shuffle_metrics = {"natural": [], "matched_budget": []}
    for seed, variant_bundles in shuffle_bundles.items():
        if seed == args.seed:
            shuffle_metrics["natural"].append(results[f"NB4_cross_label_outcome_shuffle:natural:seed={seed}"])
            shuffle_metrics["matched_budget"].append(results[f"NB4_cross_label_outcome_shuffle:matched_budget:seed={seed}"])
            continue
        scores = {key: bundle["score"] for key, bundle in variant_bundles.items()}
        for mode, selected in (
            ("natural", {key for key in ids if scores[key] > args.utility_threshold}),
            ("matched_budget", top_count(ids, scores, matched_count)),
        ):
            shuffle_metrics[mode].append(record_run(
                "NB4_cross_label_outcome_shuffle", mode, seed, selected,
                variant_bundles, save_predictions=False,
            ))

    nb0_matched_predictions = prediction_index[("NB0_full_nlf", "matched_budget", None)]
    comparison_predictions = {
        variant: prediction_index[(
            variant,
            "matched_budget",
            args.seed if variant.startswith("NB4_") else None,
        )]
        for variant in PRIMARY_VARIANTS if variant != "NB0_full_nlf"
    }
    comparisons, raw_p = {}, {}
    for variant, predictions in comparison_predictions.items():
        comparison = {
            "orientation": "NB0_minus_variant",
            "mcnemar_exact": mcnemar_exact(gold, predictions, nb0_matched_predictions),
            "paired_bootstrap": paired_bootstrap(
                gold, predictions, nb0_matched_predictions,
                args.bootstrap_samples, args.seed,
            ),
        }
        comparisons[variant] = comparison
        raw_p[variant] = comparison["mcnemar_exact"]["exact_two_sided_p"]
    for variant, value in holm_adjust(raw_p).items():
        comparisons[variant]["holm_adjusted_mcnemar_p"] = value

    primary_natural = {
        variant: {
            key for key in ids
            if all_bundles[variant][key]["score"] > args.utility_threshold
        }
        for variant in PRIMARY_VARIANTS
    }
    for key in ids:
        mechanism[key]["route_changed_vs_hardlabel"] = (
            (key in primary_natural["NB0_full_nlf"])
            != (key in primary_natural["NB1_hard_label"])
        )
        mechanism[key]["helpful_or_harmful_after_route"] = route_outcome(
            key, primary_natural["NB0_full_nlf"], baseline, candidate
        )
        mechanism[key]["natural_routes"] = {
            variant: key in selected for variant, selected in primary_natural.items()
        }

    fold_results = {}
    for variant, selected in primary_natural.items():
        fold_results[variant] = []
        for fold in range(args.folds):
            fold_ids = [key for key in ids if baseline[key]["outer_fold"] == fold]
            metrics, _ = evaluate_selection(fold_ids, baseline, candidate, selected & set(fold_ids))
            fold_results[variant].append({"outer_fold": fold, **metrics})

    cross_queries = sum(row["cross_label_count_at_5"] > 0 for row in mechanism.values())
    naturally_routed = primary_natural["NB0_full_nlf"]
    routed_cross = sum(mechanism[key]["cross_label_count_at_5"] > 0 for key in naturally_routed)
    report = {
        "evaluation": "grouped_fold_excluded_cross_label_memory_mechanism",
        "configuration": {
            "action": args.action,
            "top_k": args.top_k,
            "min_support": args.min_support,
            "cost_weight": args.cost_weight,
            "utility_threshold": args.utility_threshold,
            "soft_label_penalty": args.soft_label_penalty,
            "cross_label_quotas": [1, 2],
            "group_threshold": args.group_threshold,
            "folds": args.folds,
            "seed": args.seed,
            "shuffle_seeds": args.shuffle_seeds,
            "bootstrap_samples": args.bootstrap_samples,
            "matched_route_count": matched_count,
            "matched_route_rate": matched_count / len(ids),
        },
        "grouping": grouping,
        "baseline": base_metrics,
        "primary_results": {
            key: value for key, value in results.items()
            if "seed=" not in key or key.endswith(f"seed={args.seed}")
        },
        "nb4_stochastic_aggregate": {
            mode: summarize_replicates(rows) for mode, rows in shuffle_metrics.items()
        },
        "matched_significance_nb0_minus_variant": comparisons,
        "fold_results_natural": fold_results,
        "mechanism_summary": {
            "queries_with_cross_label_neighbor": cross_queries,
            "query_cross_label_neighbor_rate": cross_queries / len(ids),
            "naturally_routed_queries": len(naturally_routed),
            "naturally_routed_with_cross_label_neighbor": routed_cross,
            "naturally_routed_cross_label_neighbor_rate": (
                routed_cross / len(naturally_routed) if naturally_routed else 0.0
            ),
            "route_changed_vs_hardlabel": sum(
                row["route_changed_vs_hardlabel"] for row in mechanism.values()
            ),
        },
        "definitions": {
            "NB3": "Filter cross-label records from the already retrieved NLF top-k; do not backfill; insufficient remaining support yields -infinity utility.",
            "NB4": "Shuffle record outcomes within source outer-fold and source predicted-label strata, then apply replacements only when the record is cross-label for the query.",
            "utility_contributions": "Same/cross weighted (recovery-regression) numerators divided by W+2; the constant cost penalty is reported separately.",
            "NB5": "Take same-label neighbors first and use cross-label neighbors only when fewer than top-k same-label records remain.",
            "NB6": "Multiply cross-label similarities by the fixed soft-label-penalty before top-k selection.",
        },
        "integrity_notes": [
            "Routing uses no target gold label.",
            "Every lookup excludes the target claim and its grouped outer fold.",
            "NB4 seed 2026 is used for paired inference; all declared seeds are summarized for variability.",
            "NB7 quota 2 is the primary quota control; quota 1 is reported as sensitivity.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", report)
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(args.output_dir / "routes.jsonl", routes)
    write_jsonl(
        args.output_dir / "mechanistic.jsonl",
        (mechanism[key] for key in ids),
    )
    (args.output_dir / "RESULTS_FOR_CODEX.txt").write_text(
        format_results(report), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "matched_route_count": matched_count,
        "primary_results": report["primary_results"],
        "mechanism_summary": report["mechanism_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
