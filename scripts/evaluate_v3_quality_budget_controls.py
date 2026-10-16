"""Build accuracy--cost curves for fixed routers and simple controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def gold_label(row):
    return row.get("label", row.get("gold_label"))


def score_distribution(row):
    scores = row.get("label_scores") or row.get("metadata", {}).get("label_scores", {})
    if not scores:
        return {}
    peak = max(float(value) for value in scores.values())
    values = [math.exp(float(value) - peak) for value in scores.values()]
    total = sum(values)
    return {key: value / total for key, value in zip(scores, values)}


def entropy(row):
    return -sum(value * math.log(max(value, 1e-300)) for value in score_distribution(row).values())


def top_count(ids, scores, count):
    return set(sorted(ids, key=lambda key: (-scores[key], key))[:count])


def evaluate(ids, gold, baseline, candidate, selected):
    expected = [gold_label(gold[key]) for key in ids]
    before = [baseline[key]["prediction"] for key in ids]
    after = [candidate[key]["prediction"] if key in selected else baseline[key]["prediction"] for key in ids]
    helpful = sum(b != g and a == g for g, b, a in zip(expected, before, after))
    harmful = sum(b == g and a != g for g, b, a in zip(expected, before, after))
    return {
        "routed": len(selected), "route_rate": len(selected) / len(ids),
        "accuracy": accuracy(after, expected), "macro_f1": macro_f1(after, expected),
        "helpful": helpful, "harmful": harmful, "net": helpful - harmful,
        "avg_model_forward_passes": 4.0 + 4.0 * len(selected) / len(ids),
    }, after


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--meta-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0, .05, .10, .15, .20, .30, .40, .50, 1.0])
    parser.add_argument("--random-seeds", nargs="+", type=int, default=list(range(2026, 2046)))
    parser.add_argument("--project-conflicting-to-nei", action="store_true")
    args = parser.parse_args()

    gold = indexed(read_jsonl(args.gold), "gold")
    baseline = indexed(read_jsonl(args.baseline), "baseline")
    candidate = indexed(read_jsonl(args.candidate), "candidate")
    features = indexed(read_jsonl(args.meta_dir / "per_query_features.jsonl"), "meta features")
    ids = sorted(gold)
    if any(set(ids) != set(rows) for rows in (baseline, candidate, features)):
        raise ValueError("Gold, B1, A3, and meta feature IDs must match")
    if any(not 0 <= budget <= 1 for budget in args.budgets):
        raise ValueError("Budgets must be in [0, 1]")
    if args.project_conflicting_to_nei:
        baseline = {
            key: {**row, "prediction": "NOT ENOUGH INFO" if row["prediction"] == "CONFLICTING" else row["prediction"]}
            for key, row in baseline.items()
        }
        candidate = {
            key: {**row, "prediction": "NOT ENOUGH INFO" if row["prediction"] == "CONFLICTING" else row["prediction"]}
            for key, row in candidate.items()
        }

    confidence = {key: entropy(baseline[key]) for key in ids}
    def utility(key, mode):
        value = features[key][mode].get("utility")
        return float(value) if value is not None else -math.inf

    hard = {key: utility(key, "hard") for key in ids}
    nlf = {key: utility(key, "nlf") for key in ids}
    effect = {}
    for key in ids:
        expected = gold_label(gold[key])
        before = baseline[key]["prediction"] == expected
        after = candidate[key]["prediction"] == expected
        effect[key] = int(after) - int(before)

    curve_rows, prediction_rows = [], []
    for budget in sorted(set(args.budgets)):
        count = round(len(ids) * budget)
        selections = {
            "Confidence": top_count(ids, confidence, count),
            "HardLabel-memory": top_count(ids, hard, count),
            "No-label-filter-memory": top_count(ids, nlf, count),
            "Oracle-exact-budget": top_count(ids, effect, count),
        }
        helpful_ranked = sorted((key for key in ids if effect[key] > 0), key=str)
        selections["Oracle-at-most-budget"] = set(helpful_ranked[:count])
        for policy, selected in selections.items():
            result, predicted = evaluate(ids, gold, baseline, candidate, selected)
            curve_rows.append({"policy": policy, "budget": budget, **result})
            for key, value in zip(ids, predicted):
                prediction_rows.append({
                    "example_id": key, "policy": policy, "budget": budget,
                    "routed": key in selected, "gold_label": gold_label(gold[key]), "prediction": value,
                })
        random_results = []
        for seed in args.random_seeds:
            scores = {
                key: int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:16], 16)
                for key in ids
            }
            selected = top_count(ids, scores, count)
            result, _ = evaluate(ids, gold, baseline, candidate, selected)
            random_results.append(result)
        row = {"policy": "Random-mean", "budget": budget, "seeds": len(args.random_seeds)}
        for field in ("routed", "route_rate", "accuracy", "macro_f1", "helpful", "harmful", "net", "avg_model_forward_passes"):
            values = [item[field] for item in random_results]
            row[field] = statistics.mean(values)
            row[f"{field}_std"] = statistics.pstdev(values)
        curve_rows.append(row)

    natural_nlf = {
        key for key in ids
        if features[key]["nlf"].get("eligible") and nlf[key] > 0.0
    }
    natural_metrics, _ = evaluate(ids, gold, baseline, candidate, natural_nlf)
    oracle_helpful = sum(value > 0 for value in effect.values())
    summary = {
        "evaluation": "quality_budget_controls",
        "n": len(ids), "budgets": sorted(set(args.budgets)),
        "random_seeds": args.random_seeds,
        "oracle": {
            "helpful_actions_available": oracle_helpful,
            "harmful_actions_available": sum(value < 0 for value in effect.values()),
            "maximum_accuracy_gain": oracle_helpful / len(ids),
        },
        "natural_nlf": {
            **natural_metrics,
            "fraction_of_oracle_helpful_gain": natural_metrics["net"] / oracle_helpful if oracle_helpful else None,
        },
        "target_gold_used_only_for_post_hoc_oracle_and_evaluation": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "quality_budget_curves.csv", curve_rows)
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)
    atomic_write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
