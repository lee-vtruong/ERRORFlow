"""Post-hoc implementation audit for ERRORFlow V3 M1/M2/M3a artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from errorflow.io import atomic_write_json


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def decision(row, policy):
    return row["decisions"][policy]


def infer_beta(row, configs):
    if "applied_m2_beta" in row:
        return float(row["applied_m2_beta"])
    by_fold = configs.get("cross_fitted_evaluation_beta_by_outer_fold")
    if by_fold is None:
        by_fold = configs.get("beta_by_outer_fold")
    if by_fold is not None:
        fold = str(row["outer_fold"])
        if fold not in by_fold:
            raise ValueError(f"Missing beta for outer fold {fold}")
        return float(by_fold[fold])
    return float(configs["source_selected_deployable_beta"])


def prediction_index(rows):
    result = {}
    random_rows = defaultdict(dict)
    for row in rows:
        key = str(row["example_id"])
        policy = row["policy"]
        if policy == "D4_Random":
            random_rows[int(row["seed"])][key] = row
        elif row.get("mode") == "natural":
            result[(policy, key)] = row
    return result, random_rows


def rate(count, denominator):
    return count / denominator if denominator else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    directory = args.output_dir
    configs = json.loads((directory / "method_configs.json").read_text(encoding="utf-8"))
    features = list(read_jsonl(directory / "per_query_features.jsonl"))
    predictions, random_predictions = prediction_index(
        read_jsonl(directory / "per_query_predictions.jsonl")
    )
    if not features:
        raise ValueError("No per-query feature rows")

    beta_counts = Counter()
    m2_m1_mismatches = []
    zero_beta_mismatches = []
    disagreement_rows = []
    random_correct = Counter()
    random_total = Counter()
    random_route_differences = Counter()
    random_disagreement_total = Counter()
    m3_correct = union_correct = 0
    actionable = 0

    for row in features:
        key = str(row["example_id"])
        beta = infer_beta(row, configs)
        beta_counts[f"{beta:g}"] += 1
        m1 = decision(row, "M1_MaxUtility")
        m2 = decision(row, "M2_Reliability")
        hard = decision(row, "D0_HardLabel")
        nlf = decision(row, "D1_NLF")
        union = decision(row, "D2_Union")
        m3 = decision(row, "M3a_DisagreementGate")
        if bool(m1["route"]) != bool(m2["route"]):
            m2_m1_mismatches.append(key)
            if math.isclose(beta, 0.0, abs_tol=1e-15):
                zero_beta_mismatches.append(key)
        if bool(hard["route"]) == bool(nlf["route"]):
            continue

        b1 = predictions.get(("B1", key))
        a3 = predictions.get(("A3-All", key))
        if b1 is None or a3 is None:
            raise ValueError(f"Missing B1/A3 prediction for {key}")
        b1_correct = b1["prediction"] == b1["gold_label"]
        a3_correct = a3["prediction"] == a3["gold_label"]
        is_actionable = b1_correct != a3_correct
        oracle_route = a3_correct and not b1_correct
        m3_action_correct = (bool(m3["route"]) == oracle_route) if is_actionable else None
        union_action_correct = (bool(union["route"]) == oracle_route) if is_actionable else None
        if is_actionable:
            actionable += 1
            m3_correct += int(m3_action_correct)
            union_correct += int(union_action_correct)
        random_seed_correct = {}
        for seed, indexed in random_predictions.items():
            if key not in indexed:
                continue
            random_route_differences[seed] += int(
                bool(indexed[key]["routed"]) != bool(m3["route"])
            )
            random_disagreement_total[seed] += 1
            if not is_actionable:
                continue
            correct = bool(indexed[key]["routed"]) == oracle_route
            random_correct[seed] += int(correct)
            random_total[seed] += 1
            random_seed_correct[str(seed)] = correct
        disagreement_rows.append({
            "example_id": key,
            "applied_beta": beta,
            "hard_route": bool(hard["route"]),
            "nlf_route": bool(nlf["route"]),
            "union_route": bool(union["route"]),
            "m1_route": bool(m1["route"]),
            "m2_route": bool(m2["route"]),
            "m3a_route": bool(m3["route"]),
            "m3a_matches_hard": bool(m3["route"]) == bool(hard["route"]),
            "m3a_matches_nlf": bool(m3["route"]) == bool(nlf["route"]),
            "m3a_matches_union": bool(m3["route"]) == bool(union["route"]),
            "b1_correct": b1_correct,
            "a3_correct": a3_correct,
            "actionable": is_actionable,
            "oracle_route": oracle_route if is_actionable else "",
            "m3a_action_correct": m3_action_correct if is_actionable else "",
            "union_action_correct": union_action_correct if is_actionable else "",
            "m3a_oracle_regret": int(not m3_action_correct) if is_actionable else 0,
            "union_oracle_regret": int(not union_action_correct) if is_actionable else 0,
            "random_action_correct_by_seed": json.dumps(random_seed_correct, sort_keys=True),
        })

    disagreement_n = len(disagreement_rows)
    m3_hard = sum(row["m3a_matches_hard"] for row in disagreement_rows)
    m3_nlf = sum(row["m3a_matches_nlf"] for row in disagreement_rows)
    m3_union = sum(row["m3a_matches_union"] for row in disagreement_rows)
    random_rates = [
        rate(random_correct[seed], random_total[seed])
        for seed in sorted(random_total)
    ]
    random_route_difference_rates = [
        rate(random_route_differences[seed], random_disagreement_total[seed])
        for seed in sorted(random_disagreement_total)
    ]
    deployable_beta = float(configs["source_selected_deployable_beta"])
    applied_all_zero = set(beta_counts) <= {"0"}
    report = {
        "evaluation": "meta_routing_implementation_audit",
        "output_dir": str(directory),
        "P0_m2": {
            "source_selected_deployable_beta": deployable_beta,
            "applied_beta_counts": dict(beta_counts),
            "beta_protocol": (
                "cross_fitted_per_outer_fold"
                if configs.get("beta_by_outer_fold") is not None
                else "fixed_source_selected_beta"
            ),
            "m2_m1_route_mismatches": len(m2_m1_mismatches),
            "m2_m1_route_mismatch_examples": m2_m1_mismatches[:25],
            "zero_applied_beta_mismatches": len(zero_beta_mismatches),
            "zero_applied_beta_mismatch_examples": zero_beta_mismatches[:25],
            "beta0_invariant_passed": not zero_beta_mismatches,
            "interpretation": (
                "M2 is exactly M1 because beta=0 was applied to every query."
                if applied_all_zero else
                "The displayed deployable beta is not the beta applied to every grouped-internal query; "
                "evaluation used fold-specific cross-fitted betas."
            ),
        },
        "P1_m3a": {
            "disagreement_claims": disagreement_n,
            "selected_hardlabel_count": m3_hard,
            "selected_hardlabel_rate": rate(m3_hard, disagreement_n),
            "selected_nlf_count": m3_nlf,
            "selected_nlf_rate": rate(m3_nlf, disagreement_n),
            "matches_union_count": m3_union,
            "matches_union_rate": rate(m3_union, disagreement_n),
            "actionable_disagreements": actionable,
            "m3a_actionable_correct": m3_correct,
            "m3a_actionable_accuracy": rate(m3_correct, actionable),
            "union_actionable_correct": union_correct,
            "union_actionable_accuracy": rate(union_correct, actionable),
            "random_actionable_accuracy_mean": statistics.mean(random_rates) if random_rates else None,
            "random_actionable_accuracy_std": statistics.pstdev(random_rates) if len(random_rates) > 1 else 0.0,
            "m3a_minus_random_actionable_accuracy_mean": (
                rate(m3_correct, actionable) - statistics.mean(random_rates)
                if random_rates else None
            ),
            "m3a_vs_random_route_difference_rate_mean": (
                statistics.mean(random_route_difference_rates)
                if random_route_difference_rates else None
            ),
            "m3a_vs_random_route_difference_rate_std": (
                statistics.pstdev(random_route_difference_rates)
                if len(random_route_difference_rates) > 1 else 0.0
            ),
            "random_seeds": len(random_rates),
            "structural_note": (
                "With beta=0, M3a's positive reliability gate equals the routing side's original "
                "positive-utility decision, so M3a must reproduce Union in disagreements."
            ),
        },
        "M3b": {
            "implemented": False,
            "status": configs.get("M3b", "not_recorded"),
            "interpretation": "M3b is an optional trained comparator and was not evaluated.",
        },
    }
    if args.strict and zero_beta_mismatches:
        raise ValueError(f"M2 beta=0 invariant failed for {zero_beta_mismatches[:3]}")

    write_csv(directory / "meta_disagreement_audit.csv", disagreement_rows)
    atomic_write_json(directory / "implementation_audit.json", report)
    markdown = [
        "# Meta-routing implementation audit", "",
        "## P0: M2 beta semantics", "",
        f"- Deployable beta: `{deployable_beta:g}`",
        f"- Applied beta counts: `{dict(beta_counts)}`",
        f"- M2/M1 route mismatches: `{len(m2_m1_mismatches)}`",
        f"- Mismatches where the applied beta is zero: `{len(zero_beta_mismatches)}`",
        f"- Interpretation: {report['P0_m2']['interpretation']}", "",
        "## P1: M3a disagreement behavior", "",
        f"- Disagreements: `{disagreement_n}`",
        f"- Selects HardLabel: `{m3_hard}` ({rate(m3_hard, disagreement_n):.4f})",
        f"- Selects NLF: `{m3_nlf}` ({rate(m3_nlf, disagreement_n):.4f})",
        f"- Matches Union: `{m3_union}` ({rate(m3_union, disagreement_n):.4f})",
        f"- Actionable disagreements: `{actionable}`",
        f"- M3a actionable accuracy: `{rate(m3_correct, actionable):.4f}`",
        f"- Union actionable accuracy: `{rate(union_correct, actionable):.4f}`",
        f"- Random actionable accuracy mean: `{report['P1_m3a']['random_actionable_accuracy_mean']}`", "",
        f"- M3a minus random actionable accuracy: `{report['P1_m3a']['m3a_minus_random_actionable_accuracy_mean']}`",
        f"- M3a/random route difference rate: `{report['P1_m3a']['m3a_vs_random_route_difference_rate_mean']}`", "",
        "## M3b status", "",
        "M3b is not implemented or evaluated in this benchmark.", "",
    ]
    (directory / "implementation_audit.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"META_ROUTING_AUDIT_DONE output_dir={directory}")


if __name__ == "__main__":
    main()
