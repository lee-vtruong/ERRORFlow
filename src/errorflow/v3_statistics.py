"""Dependency-free paired statistics for V3 comparisons."""

import math
import random

from .evaluation_v2 import accuracy, macro_f1


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def mcnemar_exact(gold, baseline, candidate):
    baseline_only = sum(b == g and c != g for g, b, c in zip(gold, baseline, candidate))
    candidate_only = sum(b != g and c == g for g, b, c in zip(gold, baseline, candidate))
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(0, min(baseline_only, candidate_only) + 1)) / (2 ** discordant)
        p_value = min(1.0, 2 * tail)
    return {"baseline_only_correct": baseline_only, "candidate_only_correct": candidate_only,
            "discordant": discordant, "exact_two_sided_p": p_value}


def paired_bootstrap(gold, baseline, candidate, samples=5000, seed=2026):
    rng = random.Random(seed)
    accuracy_deltas, f1_deltas = [], []
    for _ in range(samples):
        indexes = [rng.randrange(len(gold)) for _ in gold]
        sampled_gold = [gold[index] for index in indexes]
        sampled_baseline = [baseline[index] for index in indexes]
        sampled_candidate = [candidate[index] for index in indexes]
        accuracy_deltas.append(accuracy(sampled_candidate, sampled_gold) - accuracy(sampled_baseline, sampled_gold))
        f1_deltas.append(macro_f1(sampled_candidate, sampled_gold) - macro_f1(sampled_baseline, sampled_gold))
    return {
        "samples": samples, "seed": seed,
        "accuracy_delta_ci95": [percentile(accuracy_deltas, 0.025), percentile(accuracy_deltas, 0.975)],
        "macro_f1_delta_ci95": [percentile(f1_deltas, 0.025), percentile(f1_deltas, 0.975)],
        "p_delta_accuracy_gt_0": sum(value > 0 for value in accuracy_deltas) / samples,
        "p_delta_macro_f1_gt_0": sum(value > 0 for value in f1_deltas) / samples,
    }
