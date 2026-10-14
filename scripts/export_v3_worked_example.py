"""Export two genuine fold-safe R6-NLF versus HardLabel worked traces.

This script performs no model inference.  It recomputes the frozen grouped
memory lookup and selects post-hoc qualitative examples from cached B1/A3
predictions.  Case selection is explicitly outcome-aware and therefore must
never be counted as quantitative evaluation or model selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl
from errorflow.v3_memory import ResidualMemoryStore
from errorflow.v3_routing import estimate_action

try:
    from scripts.evaluate_v3_cross_label_ablation import estimate_bundle, query_for
    from scripts.evaluate_v3_matched_budget import indexed, prepare_grouped_artifacts
    from scripts.evaluate_v3_memory_ablation import FULL_WEIGHTS
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    from evaluate_v3_cross_label_ablation import estimate_bundle, query_for
    from evaluate_v3_matched_budget import indexed, prepare_grouped_artifacts
    from evaluate_v3_memory_ablation import FULL_WEIGHTS


CASE_ORDER = (
    "nlf_prevents_harmful_hardlabel_correction",
    "nlf_helpful_correction_missed_by_hardlabel",
)


def finite_or_none(value):
    return value if math.isfinite(value) else None


def route(score, threshold):
    return math.isfinite(score) and score > threshold


def case_kind(gold, before, after, nlf_routes, hard_routes):
    if before == gold and after != gold and hard_routes and not nlf_routes:
        return CASE_ORDER[0]
    if before != gold and after == gold and nlf_routes and not hard_routes:
        return CASE_ORDER[1]
    return None


def estimate_payload(neighbors, action, query_label, min_support, cost_weight):
    bundle = estimate_bundle(neighbors, action, query_label, min_support, cost_weight)
    estimate = estimate_action(action, neighbors, cost_weight)
    return {
        "route_utility": finite_or_none(bundle["score"]),
        "raw_utility_before_support_gate": finite_or_none(estimate.utility),
        "support": bundle["support"],
        "raw_support": estimate.support,
        "expected_recovery": estimate.expected_recovery,
        "expected_regression": estimate.expected_regression,
        "expected_cost": estimate.expected_cost,
        "same_label_utility_contribution": bundle["same_label_utility_contribution"],
        "cross_label_utility_contribution": bundle["cross_label_utility_contribution"],
        "cross_label_count_at_5": bundle["cross_label_count_at_5"],
        "cross_label_similarity_mass": bundle["cross_label_similarity_mass"],
    }


def neighbor_payload(neighbors, query_label):
    output = []
    for rank, (record, similarity) in enumerate(neighbors, 1):
        source_label = record.retrieval_features.get("baseline_prediction")
        output.append({
            "rank": rank,
            "memory_id": record.memory_id,
            "source_claim_id": record.source_claim_id,
            "source_claim": record.source_claim,
            "source_baseline_prediction": source_label,
            "same_label_as_query": source_label == query_label,
            "similarity": similarity,
            "recovery_count": record.recovery_count,
            "regression_count": record.regression_count,
            "observed_outcome": (
                "helpful" if record.recovery_count else
                "harmful" if record.regression_count else "neutral"
            ),
            "outer_fold": record.provenance.get("outer_fold"),
        })
    return output


def md(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report):
    lines = [
        "# ERRORFlow V3 worked case studies",
        "",
        "> Post-hoc qualitative examples exported from fold-excluded artifacts. "
        "They are not additional evaluation samples and were selected using gold correctness only for explanation.",
        "",
    ]
    for kind in CASE_ORDER:
        case = report["cases"].get(kind)
        lines.append(f"## {kind.replace('_', ' ').title()}")
        lines.append("")
        if case is None:
            lines.extend(("No qualifying case was found at the frozen natural threshold.", ""))
            continue
        lines.extend((
            f"- Example: `{md(case['example_id'])}`; outer fold: {case['outer_fold']}",
            f"- Claim: {md(case['claim'])}",
            f"- Gold / B1 / A3: **{case['gold_label']}** / "
            f"**{case['baseline_prediction']}** / **{case['candidate_prediction']}**",
            f"- Natural route: NLF={case['nlf']['routed']}, HardLabel={case['hardlabel']['routed']}",
            f"- Utility: NLF={case['nlf']['estimate']['route_utility']}, "
            f"HardLabel={case['hardlabel']['estimate']['route_utility']}",
            "",
            "Evidence:",
            "",
        ))
        for index, evidence in enumerate(case["evidence"], 1):
            text = evidence.get("text", "") if isinstance(evidence, dict) else str(evidence)
            lines.append(f"{index}. {md(text)}")
        for router in ("nlf", "hardlabel"):
            lines.extend(("", f"### {router} neighbors", "", "| Rank | Label | Same? | Similarity | Outcome | Source claim |", "|---:|---|:---:|---:|---|---|"))
            for row in case[router]["neighbors"]:
                lines.append(
                    f"| {row['rank']} | {md(row['source_baseline_prediction'])} | "
                    f"{row['same_label_as_query']} | {row['similarity']:.4f} | "
                    f"{row['observed_outcome']} | {md(row['source_claim'])} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--action", default="conflicting_rescue_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--group-threshold", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    example_rows = list(read_jsonl(args.examples))
    baseline_rows = list(read_jsonl(args.baseline_records))
    candidate_rows = list(read_jsonl(args.candidate))
    memory_rows = list(read_jsonl(args.memory))
    baseline_rows, records, _, grouping = prepare_grouped_artifacts(
        baseline_rows, memory_rows, args.group_threshold, args.folds, args.seed,
    )
    examples = indexed(example_rows, "examples")
    baseline = indexed(baseline_rows, "baseline")
    candidate = indexed(candidate_rows, "candidate")
    if set(baseline) != set(candidate) or not set(baseline) <= set(examples):
        raise ValueError("Examples, baseline, and candidate IDs are not aligned")

    store = ResidualMemoryStore(records)
    eligible = {kind: [] for kind in CASE_ORDER}
    for key in sorted(baseline):
        row = baseline[key]
        common = {
            "top_k": args.top_k,
            "action": args.action,
            "exclude_claim_ids": {key},
            "exclude_folds": {row["outer_fold"]},
            "similarity_weights": FULL_WEIGHTS,
        }
        query = query_for(key, row)
        nlf_neighbors = store.retrieve(query, compatibility_mode="none", **common)
        hard_neighbors = store.retrieve(query, compatibility_mode="same_label", **common)
        nlf = estimate_payload(
            nlf_neighbors, args.action, row["prediction"], args.min_support, args.cost_weight
        )
        hard = estimate_payload(
            hard_neighbors, args.action, row["prediction"], args.min_support, args.cost_weight
        )
        nlf_routes = route(nlf["route_utility"] if nlf["route_utility"] is not None else -math.inf, args.utility_threshold)
        hard_routes = route(hard["route_utility"] if hard["route_utility"] is not None else -math.inf, args.utility_threshold)
        kind = case_kind(
            row["gold_label"], row["prediction"], candidate[key]["prediction"],
            nlf_routes, hard_routes,
        )
        if kind is None:
            continue
        nlf_score = nlf["route_utility"] if nlf["route_utility"] is not None else -math.inf
        hard_score = hard["route_utility"] if hard["route_utility"] is not None else -math.inf
        gap = (hard_score - nlf_score) if kind == CASE_ORDER[0] else (nlf_score - hard_score)
        eligible[kind].append((
            gap,
            nlf["cross_label_count_at_5"],
            key,
            {
                "case_type": kind,
                "selection_status": "post_hoc_gold_aware_qualitative_only",
                "example_id": key,
                "outer_fold": row["outer_fold"],
                "claim": examples[key]["claim"],
                "evidence": examples[key].get("evidence", []),
                "gold_label": row["gold_label"],
                "baseline_prediction": row["prediction"],
                "candidate_prediction": candidate[key]["prediction"],
                "utility_gap_in_favor_of_observed_route_decision": gap,
                "nlf": {
                    "routed": nlf_routes,
                    "estimate": nlf,
                    "neighbors": neighbor_payload(nlf_neighbors, row["prediction"]),
                },
                "hardlabel": {
                    "routed": hard_routes,
                    "estimate": hard,
                    "neighbors": neighbor_payload(hard_neighbors, row["prediction"]),
                },
            },
        ))

    cases = {}
    eligible_counts = {}
    for kind, rows in eligible.items():
        eligible_counts[kind] = len(rows)
        rows.sort(key=lambda item: (-item[0], -item[1], item[2]))
        cases[kind] = rows[0][3] if rows else None
    missing = [kind for kind, case in cases.items() if case is None]
    report = {
        "evaluation": "post_hoc_fold_safe_worked_case_export",
        "configuration": vars(args) | {
            "examples": str(args.examples),
            "baseline_records": str(args.baseline_records),
            "candidate": str(args.candidate),
            "memory": str(args.memory),
            "output_dir": str(args.output_dir),
        },
        "grouping": grouping,
        "selection_rule": {
            CASE_ORDER[0]: "B1 correct, A3 wrong, HardLabel routes, NLF abstains; choose largest utility separation.",
            CASE_ORDER[1]: "B1 wrong, A3 correct, NLF routes, HardLabel abstains; choose largest utility separation.",
        },
        "eligible_counts": eligible_counts,
        "missing_case_types": missing,
        "cases": cases,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "case_studies.json", report)
    (args.output_dir / "case_studies.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "eligible_counts": eligible_counts,
        "selected_ids": {kind: case["example_id"] if case else None for kind, case in cases.items()},
    }, indent=2))
    if missing and not args.allow_missing:
        raise SystemExit(
            "Missing required case type(s): " + ", ".join(missing) +
            ". Artifacts were still written; use --allow-missing only to accept this explicitly."
        )


if __name__ == "__main__":
    main()
