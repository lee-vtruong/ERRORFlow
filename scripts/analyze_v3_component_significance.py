"""Paired significance and fold analysis for cached R6 component ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap

try:
    from scripts.evaluate_v3_matched_budget import holm_adjust, indexed, prepare_grouped_artifacts
except ModuleNotFoundError:  # Direct `python scripts/...py` execution.
    from evaluate_v3_matched_budget import holm_adjust, indexed, prepare_grouped_artifacts


PRIMARY_VARIANTS = [
    "A1_without_claim",
    "A2_without_label_score",
    "A3_without_evidence_length",
    "A4_label_score_only",
    "A5_claim_only",
    "A6_evidence_length_only",
    "A7_uniform_weights",
    "A8_random_neighbors",
]


def materialize_predictions(ids, baseline, candidate, selected):
    return [candidate[key]["prediction"] if key in selected else baseline[key]["prediction"] for key in ids]


def pair_report(gold, variant_predictions, a0_predictions, samples=5000, seed=2026,
                include_bootstrap=True):
    test = mcnemar_exact(gold, variant_predictions, a0_predictions)
    report = {
        "n": len(gold),
        "variant_only_correct": test["baseline_only_correct"],
        "a0_only_correct": test["candidate_only_correct"],
        "discordant": test["discordant"],
        "mcnemar_exact_two_sided_p": test["exact_two_sided_p"],
        "a0_minus_variant_accuracy": accuracy(a0_predictions, gold) - accuracy(variant_predictions, gold),
        "a0_minus_variant_macro_f1": macro_f1(a0_predictions, gold) - macro_f1(variant_predictions, gold),
    }
    if include_bootstrap:
        report["paired_bootstrap_a0_minus_variant"] = paired_bootstrap(
            gold, variant_predictions, a0_predictions, samples, seed,
        )
    return report


def fold_report(ids, folds, gold_by_id, variant_by_id, a0_by_id):
    rows = []
    for fold in sorted(set(folds.values())):
        members = [key for key in ids if folds[key] == fold]
        gold = [gold_by_id[key] for key in members]
        variant = [variant_by_id[key] for key in members]
        a0 = [a0_by_id[key] for key in members]
        test = mcnemar_exact(gold, variant, a0)
        rows.append({
            "outer_fold": fold,
            "n": len(members),
            "variant_accuracy": accuracy(variant, gold),
            "a0_accuracy": accuracy(a0, gold),
            "a0_minus_variant_accuracy": accuracy(a0, gold) - accuracy(variant, gold),
            "variant_macro_f1": macro_f1(variant, gold),
            "a0_macro_f1": macro_f1(a0, gold),
            "a0_minus_variant_macro_f1": macro_f1(a0, gold) - macro_f1(variant, gold),
            "variant_only_correct": test["baseline_only_correct"],
            "a0_only_correct": test["candidate_only_correct"],
        })
    return {
        "folds": rows,
        "a0_better_accuracy_folds": sum(row["a0_minus_variant_accuracy"] > 0 for row in rows),
        "a0_better_macro_f1_folds": sum(row["a0_minus_variant_macro_f1"] > 0 for row in rows),
        "variant_better_accuracy_folds": sum(row["a0_minus_variant_accuracy"] < 0 for row in rows),
        "variant_better_macro_f1_folds": sum(row["a0_minus_variant_macro_f1"] < 0 for row in rows),
    }


def route_index(rows):
    result = {}
    for row in rows:
        key = (row["variant"], row["mode"], row.get("seed"))
        if key in result:
            raise ValueError(f"Duplicate route row: {key}")
        result[key] = set(map(str, row["selected_ids"]))
    return result


def mean_range(values):
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--component-summary", required=True, type=Path)
    parser.add_argument("--component-routes", required=True, type=Path)
    parser.add_argument("--component-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--a8-primary-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    component = json.loads(args.component_summary.read_text(encoding="utf-8"))
    config = component["configuration"]
    baseline_rows = list(read_jsonl(args.baseline_records))
    memory_rows = list(read_jsonl(args.memory))
    baseline_rows, _, assignments, grouping = prepare_grouped_artifacts(
        baseline_rows,
        memory_rows,
        config["group_threshold"],
        config["folds"],
        config["seed"],
    )
    baseline = indexed(baseline_rows, "baseline")
    candidate = indexed(read_jsonl(args.candidate), "candidate")
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate IDs must match exactly")
    ids = sorted(baseline)
    gold_by_id = {key: baseline[key]["gold_label"] for key in ids}
    gold = [gold_by_id[key] for key in ids]
    routes = route_index(read_jsonl(args.component_routes))

    cached_predictions = {}
    for row in read_jsonl(args.component_predictions):
        key = (row["variant"], row["mode"], row.get("seed"), str(row["example_id"]))
        if key in cached_predictions:
            raise ValueError(f"Duplicate component prediction: {key}")
        cached_predictions[key] = row["prediction"]

    output_rows, families, replay_checks = [], {}, {"checked": 0, "mismatches": 0}
    for mode in ("natural", "matched_budget"):
        a0_key = ("A0_full_r6", mode, None)
        if a0_key not in routes:
            raise ValueError(f"Missing A0 route row for {mode}")
        a0_selected = routes[a0_key]
        a0_predictions = materialize_predictions(ids, baseline, candidate, a0_selected)
        a0_by_id = dict(zip(ids, a0_predictions))
        reports, raw_p = {}, {}
        for variant in PRIMARY_VARIANTS:
            seed = args.a8_primary_seed if variant == "A8_random_neighbors" else None
            key = (variant, mode, seed)
            if key not in routes:
                raise ValueError(f"Missing route row: {key}")
            selected = routes[key]
            predictions = materialize_predictions(ids, baseline, candidate, selected)
            variant_by_id = dict(zip(ids, predictions))
            pair = pair_report(gold, predictions, a0_predictions, args.bootstrap_samples, config["seed"])
            pair["outer_fold_analysis"] = fold_report(
                ids, assignments, gold_by_id, variant_by_id, a0_by_id,
            )
            pair["variant_seed"] = seed
            pair["variant_route_count"] = len(selected)
            pair["a0_route_count"] = len(a0_selected)
            reports[variant] = pair
            raw_p[variant] = pair["mcnemar_exact_two_sided_p"]
            for claim_id, a0_prediction, variant_prediction in zip(ids, a0_predictions, predictions):
                cache_key = (variant, mode, seed, claim_id)
                if cache_key in cached_predictions:
                    replay_checks["checked"] += 1
                    replay_checks["mismatches"] += cached_predictions[cache_key] != variant_prediction
                output_rows.append({
                    "example_id": claim_id,
                    "outer_fold": assignments[claim_id],
                    "mode": mode,
                    "variant": variant,
                    "variant_seed": seed,
                    "gold_label": gold_by_id[claim_id],
                    "a0_prediction": a0_prediction,
                    "variant_prediction": variant_prediction,
                    "a0_routed": claim_id in a0_selected,
                    "variant_routed": claim_id in selected,
                })
        adjusted = holm_adjust(raw_p)
        for variant in reports:
            reports[variant]["holm_adjusted_mcnemar_p"] = adjusted[variant]
            reports[variant]["holm_family_size"] = len(reports)
        families[mode] = reports

    a8_sensitivity = {}
    for mode in ("natural", "matched_budget"):
        a0_selected = routes[("A0_full_r6", mode, None)]
        a0_predictions = materialize_predictions(ids, baseline, candidate, a0_selected)
        rows = []
        for seed in config["random_neighbor_seeds"]:
            selected = routes[("A8_random_neighbors", mode, seed)]
            predictions = materialize_predictions(ids, baseline, candidate, selected)
            report = pair_report(
                gold, predictions, a0_predictions, args.bootstrap_samples,
                config["seed"], include_bootstrap=False,
            )
            rows.append({"seed": seed, **report})
        a8_sensitivity[mode] = {
            "primary_seed": args.a8_primary_seed,
            "seeds": rows,
            "a0_minus_a8_accuracy": mean_range([row["a0_minus_variant_accuracy"] for row in rows]),
            "a0_minus_a8_macro_f1": mean_range([row["a0_minus_variant_macro_f1"] for row in rows]),
            "a0_only_correct": mean_range([row["a0_only_correct"] for row in rows]),
            "a8_only_correct": mean_range([row["variant_only_correct"] for row in rows]),
        }

    report = {
        "evaluation": "paired_component_ablation_significance",
        "comparison_direction": "A0 Full R6 minus each variant",
        "holm_families": "one family of eight A1--A8 comparisons per evaluation mode",
        "a8_primary_in_holm": f"seed {args.a8_primary_seed}; remaining seeds are sensitivity only",
        "configuration": config,
        "grouping": grouping,
        "prediction_replay": replay_checks,
        "families": families,
        "a8_seed_sensitivity": a8_sensitivity,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "significance.json", report)
    write_jsonl(args.output_dir / "paired_predictions.jsonl", output_rows)
    print(json.dumps({
        "prediction_replay": replay_checks,
        "families": families,
        "a8_seed_sensitivity_summary": {
            mode: {key: value for key, value in data.items() if key != "seeds"}
            for mode, data in a8_sensitivity.items()
        },
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
