"""Metrics for baseline-to-intervention transitions."""


def transition_metrics(baseline: list[str], final: list[str], gold: list[str]) -> dict[str, float]:
    if not (len(baseline) == len(final) == len(gold)):
        raise ValueError("baseline, final and gold must have equal length")
    wrong = [b != g for b, g in zip(baseline, gold)]
    right = [b == g for b, g in zip(baseline, gold)]
    recovered = sum(b != g and f == g for b, f, g in zip(baseline, final, gold))
    regressed = sum(b == g and f != g for b, f, g in zip(baseline, final, gold))
    return {
        "n": float(len(gold)),
        "recovery_rate": recovered / sum(wrong) if any(wrong) else 0.0,
        "regression_rate": regressed / sum(right) if any(right) else 0.0,
        "baseline_accuracy": sum(right) / len(gold) if gold else 0.0,
        "final_accuracy": sum(f == g for f, g in zip(final, gold)) / len(gold) if gold else 0.0,
    }
