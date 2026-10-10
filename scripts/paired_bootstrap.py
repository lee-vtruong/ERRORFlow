"""Paired bootstrap comparison for two prediction artifacts."""
import argparse
import json
import random
from pathlib import Path

LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO")


def load(path):
    return {str(json.loads(line)["claim_id"]): json.loads(line) for line in path.open(encoding="utf-8") if line.strip()}


def metrics(rows):
    correct = sum(prediction == gold for gold, prediction in rows)
    f1s = []
    for label in LABELS:
        tp = sum(gold == label and prediction == label for gold, prediction in rows)
        fp = sum(gold != label and prediction == label for gold, prediction in rows)
        fn = sum(gold == label and prediction != label for gold, prediction in rows)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return correct / len(rows), sum(f1s) / len(f1s)


def percentile(values, probability):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    gold, baseline, candidate = load(args.gold), load(args.baseline), load(args.candidate)
    ids = sorted(set(gold) & set(baseline) & set(candidate))
    base_rows = [(gold[i]["gold_label"], baseline[i]["prediction"]) for i in ids]
    candidate_rows = [(gold[i]["gold_label"], candidate[i]["prediction"]) for i in ids]
    base_metrics, candidate_metrics = metrics(base_rows), metrics(candidate_rows)
    helpful = sum(b != g and c == g for (g, b), (_, c) in zip(base_rows, candidate_rows))
    harmful = sum(b == g and c != g for (g, b), (_, c) in zip(base_rows, candidate_rows))
    rng = random.Random(args.seed)
    accuracy_deltas, macro_f1_deltas = [], []
    for _ in range(args.samples):
        indexes = [rng.randrange(len(ids)) for _ in ids]
        sampled_base = [base_rows[i] for i in indexes]
        sampled_candidate = [candidate_rows[i] for i in indexes]
        bm, cm = metrics(sampled_base), metrics(sampled_candidate)
        accuracy_deltas.append(cm[0] - bm[0])
        macro_f1_deltas.append(cm[1] - bm[1])
    result = {
        "n": len(ids), "samples": args.samples, "seed": args.seed,
        "helpful": helpful, "harmful": harmful,
        "baseline_accuracy": base_metrics[0], "candidate_accuracy": candidate_metrics[0],
        "delta_accuracy": candidate_metrics[0] - base_metrics[0],
        "accuracy_ci95": [percentile(accuracy_deltas, 0.025), percentile(accuracy_deltas, 0.975)],
        "p_delta_accuracy_gt_0": sum(x > 0 for x in accuracy_deltas) / len(accuracy_deltas),
        "baseline_macro_f1": base_metrics[1], "candidate_macro_f1": candidate_metrics[1],
        "delta_macro_f1": candidate_metrics[1] - base_metrics[1],
        "macro_f1_ci95": [percentile(macro_f1_deltas, 0.025), percentile(macro_f1_deltas, 0.975)],
        "p_delta_macro_f1_gt_0": sum(x > 0 for x in macro_f1_deltas) / len(macro_f1_deltas),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
