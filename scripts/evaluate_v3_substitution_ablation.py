"""Evaluate ERRORFlow V3 Group-C substitution ablations from cached B1/A3.

This driver never calls the verifier.  It replays the frozen baseline and
corrective-action predictions while changing one routing mechanism at a time.
All memory retrieval excludes the target example and its grouped outer fold.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap
from errorflow.v3_substitution import (
    aggregate_similarity,
    bm25_prepare,
    bm25_score,
    build_idf,
    confidence_bin,
    estimate_utility,
    evidence_length_similarity,
    jaccard,
    label_similarity,
    margin_bin,
    probabilities,
    saturate,
    scalar_similarity,
    sparse_cosine,
    tfidf_vector,
)

try:
    from scripts.evaluate_v3_matched_budget import (
        evaluate_selection,
        holm_adjust,
        indexed,
        prepare_grouped_artifacts,
    )
except ModuleNotFoundError:
    from evaluate_v3_matched_budget import evaluate_selection, holm_adjust, indexed, prepare_grouped_artifacts


REFERENCE = {
    "claim_metric": "jaccard", "label_metric": "l1", "evidence_metric": "length",
    "aggregation": "weighted_sum", "weights": (0.35, 0.50, 0.15),
    "top_k": 5, "min_support": 2, "estimator": "smoothed",
    "compatibility": "same_label",
}


def variant(**changes):
    result = dict(REFERENCE)
    result.update(changes)
    return result


VARIANTS = {
    "C0_full_r6_reference": variant(),
    "C1-0_token_jaccard": variant(),
    "C1-1_tfidf_cosine": variant(claim_metric="tfidf"),
    "C1-2_bm25": variant(claim_metric="bm25"),
    "C1-3_sentence_bert": variant(claim_metric="sbert"),
    "C1-4_e5": variant(claim_metric="e5"),
    "C2-0_normalized_l1": variant(),
    "C2-1_cosine": variant(label_metric="cosine"),
    "C2-2_l2": variant(label_metric="l2"),
    "C2-3_jensen_shannon": variant(label_metric="jensen_shannon"),
    "C2-4_hellinger": variant(label_metric="hellinger"),
    "C3-0_evidence_length": variant(),
    "C3-1_evidence_tfidf": variant(evidence_metric="tfidf"),
    "C3-2_claim_evidence_overlap": variant(evidence_metric="claim_evidence_overlap"),
    "C3-3_evidence_bm25": variant(evidence_metric="bm25"),
    "C3-4_evidence_semantic": variant(evidence_metric="sbert"),
    "C4-0_weighted_sum": variant(),
    "C4-1_uniform_sum": variant(weights=(1.0, 1.0, 1.0)),
    "C4-2_reciprocal_rank_fusion": variant(aggregation="rrf"),
    "C4-3_multiplicative": variant(aggregation="multiplicative"),
    "C5-0_top5": variant(),
    "C5-1_top1": variant(top_k=1, min_support=1),
    "C5-2_top3": variant(top_k=3),
    "C5-3_top10": variant(top_k=10),
    "C5-4_top20": variant(top_k=20),
    "C6-0_smoothed": variant(),
    "C6-1_unsmoothed": variant(estimator="unsmoothed"),
    "C6-2_bayesian_shrunk": variant(estimator="bayesian"),
    "C6-3_effective_sample_size": variant(estimator="effective_sample_size"),
    "C7-0_same_predicted_label": variant(),
    "C7-1_no_label_filter": variant(compatibility="none"),
    "C7-2_label_and_confidence_bin": variant(compatibility="label_confidence_bin"),
    "C7-3_soft_label_compatibility": variant(compatibility="soft_label"),
    "C7-4_label_and_margin_bin": variant(compatibility="label_margin_bin"),
}


GROUPS = {
    "C1": [name for name in VARIANTS if name.startswith("C1-")],
    "C2": [name for name in VARIANTS if name.startswith("C2-")],
    "C3": [name for name in VARIANTS if name.startswith("C3-")],
    "C4": [name for name in VARIANTS if name.startswith("C4-")],
    "C5": [name for name in VARIANTS if name.startswith("C5-")],
    "C6": [name for name in VARIANTS if name.startswith("C6-")],
    "C7": [name for name in VARIANTS if name.startswith("C7-")],
}


def evidence_text(row):
    return "\n".join(
        item.get("text", "") if isinstance(item, dict) else str(item)
        for item in row.get("evidence", [])
    )


def load_dense_cache(path, expected_ids, expected_kind):
    if path is None:
        return None, None
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Dense substitutions require numpy") from error
    payload = np.load(path, allow_pickle=False)
    ids = [str(value) for value in payload["ids"].tolist()]
    if ids != expected_ids:
        raise ValueError(f"Dense cache IDs do not match baseline ordering: {path}")
    manifest_path = Path(str(path) + ".manifest.json")
    if not manifest_path.exists():
        raise ValueError(f"Missing dense-cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("embedding_kind") != expected_kind:
        raise ValueError(f"Expected embedding_kind={expected_kind}, got {manifest.get('embedding_kind')}")
    return payload["claim"], payload["evidence"]


class Engine:
    def __init__(self, ids, baseline, records, examples, action, cost_weight,
                 sbert_cache=None, e5_cache=None):
        self.ids, self.baseline, self.action, self.cost_weight = ids, baseline, action, cost_weight
        self.records = [record for record in records if record.correction_strategy == action]
        self.examples = examples
        self.position = {key: index for index, key in enumerate(ids)}
        self.record_by_id = {record.source_claim_id: record for record in self.records}
        missing = sorted(set(self.record_by_id) - set(baseline))
        if missing:
            raise ValueError(f"Memory contains IDs absent from baseline: {missing[:3]}")
        self.claims = {key: baseline[key]["claim"] for key in ids}
        self.evidence = {key: evidence_text(examples[key]) for key in ids}
        self.claim_idf = build_idf(self.claims.values())
        self.evidence_idf = build_idf(self.evidence.values())
        self.claim_tfidf = {key: tfidf_vector(text, self.claim_idf) for key, text in self.claims.items()}
        self.evidence_tfidf = {key: tfidf_vector(text, self.evidence_idf) for key, text in self.evidence.items()}
        claim_order = [self.claims[key] for key in ids]
        evidence_order = [self.evidence[key] for key in ids]
        claim_bm25 = bm25_prepare(claim_order)
        evidence_bm25 = bm25_prepare(evidence_order)
        self.claim_bm25_counts = {key: claim_bm25[0][index] for index, key in enumerate(ids)}
        self.claim_bm25_meta = claim_bm25[1:]
        self.evidence_bm25_counts = {key: evidence_bm25[0][index] for index, key in enumerate(ids)}
        self.evidence_bm25_meta = evidence_bm25[1:]
        self.overlap = {key: jaccard(self.claims[key], self.evidence[key]) for key in ids}
        self.sbert_claim, self.sbert_evidence = load_dense_cache(sbert_cache, ids, "sentence_bert")
        self.e5_claim, _ = load_dense_cache(e5_cache, ids, "e5")
        self.by_fold = defaultdict(list)
        for record in self.records:
            self.by_fold[record.provenance.get("outer_fold")].append(record)

    def dense_similarity(self, left_id, right_id, matrix):
        if matrix is None:
            raise ValueError("Requested dense variant but its embedding cache was not supplied")
        import numpy as np
        left, right = matrix[self.position[left_id]], matrix[self.position[right_id]]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        cosine = float(np.dot(left, right) / denominator) if denominator else 0.0
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    def compatible(self, query_id, record, mode):
        query = self.baseline[query_id]
        features = record.retrieval_features
        same_label = features.get("baseline_prediction") == query["prediction"]
        if mode == "none" or mode == "soft_label":
            return True
        if not same_label:
            return False
        if mode == "same_label":
            return True
        query_prob = max(probabilities(query.get("label_scores", {})) or [0.0])
        record_prob = max(probabilities(features.get("label_scores", {})) or [0.0])
        if mode == "label_confidence_bin":
            return confidence_bin(query_prob) == confidence_bin(record_prob)
        if mode == "label_margin_bin":
            return margin_bin(query.get("prediction_margin")) == margin_bin(features.get("prediction_margin"))
        raise ValueError(f"Unknown compatibility mode: {mode}")

    def component_scores(self, query_id, record, config):
        record_id = record.source_claim_id
        query, features = self.baseline[query_id], record.retrieval_features
        claim_metric = config["claim_metric"]
        if claim_metric == "jaccard":
            claim_score = jaccard(self.claims[query_id], self.claims[record_id])
        elif claim_metric == "tfidf":
            claim_score = sparse_cosine(self.claim_tfidf[query_id], self.claim_tfidf[record_id])
        elif claim_metric == "bm25":
            claim_score = saturate(bm25_score(
                self.claims[query_id], self.claim_bm25_counts[record_id], *self.claim_bm25_meta,
            ))
        elif claim_metric == "sbert":
            claim_score = self.dense_similarity(query_id, record_id, self.sbert_claim)
        elif claim_metric == "e5":
            claim_score = self.dense_similarity(query_id, record_id, self.e5_claim)
        else:
            raise ValueError(f"Unknown claim metric: {claim_metric}")
        label_score = label_similarity(query.get("label_scores", {}), features.get("label_scores", {}), config["label_metric"])
        evidence_metric = config["evidence_metric"]
        if evidence_metric == "length":
            evidence_score = evidence_length_similarity(query.get("evidence_tokens", 0), features.get("evidence_tokens", 0))
        elif evidence_metric == "tfidf":
            evidence_score = sparse_cosine(self.evidence_tfidf[query_id], self.evidence_tfidf[record_id])
        elif evidence_metric == "claim_evidence_overlap":
            evidence_score = scalar_similarity(self.overlap[query_id], self.overlap[record_id])
        elif evidence_metric == "bm25":
            evidence_score = saturate(bm25_score(
                self.evidence[query_id], self.evidence_bm25_counts[record_id], *self.evidence_bm25_meta,
            ))
        elif evidence_metric == "sbert":
            evidence_score = self.dense_similarity(query_id, record_id, self.sbert_evidence)
        else:
            raise ValueError(f"Unknown evidence metric: {evidence_metric}")
        if config["compatibility"] == "soft_label" and features.get("baseline_prediction") != query["prediction"]:
            label_score *= 0.5
        return claim_score, label_score, evidence_score

    def prior(self, query_id, candidates):
        support = max(1, len(candidates))
        return {
            "recovery": sum(record.recovery_count for record in candidates) / support,
            "regression": sum(record.regression_count for record in candidates) / support,
            "strength": 2.0,
        }

    def query_score(self, query_id, config, extra_excluded_folds=()):
        fold = self.baseline[query_id]["outer_fold"]
        excluded_folds = {fold, *extra_excluded_folds}
        candidates = [
            record for record in self.records
            if record.source_claim_id != query_id
            and record.provenance.get("outer_fold") not in excluded_folds
            and self.compatible(query_id, record, config["compatibility"])
        ]
        if not candidates:
            return float("-inf"), 0
        rows = [(record, self.component_scores(query_id, record, config)) for record in candidates]
        if config["aggregation"] == "rrf":
            ranks = []
            for component in range(3):
                ordered = sorted(rows, key=lambda item: (-item[1][component], item[0].memory_id))
                ranks.append({record.memory_id: rank for rank, (record, _) in enumerate(ordered, 1)})
            scored = [
                (record, sum(1.0 / (60.0 + rank[record.memory_id]) for rank in ranks) / (3.0 / 61.0))
                for record, _ in rows
            ]
        else:
            scored = [
                (record, aggregate_similarity(parts, config["weights"], config["aggregation"]))
                for record, parts in rows
            ]
        neighbors = heapq.nsmallest(
            config["top_k"], scored, key=lambda item: (-item[1], item[0].memory_id),
        )
        if len(neighbors) < config["min_support"]:
            return float("-inf"), len(neighbors)
        estimate = estimate_utility(
            neighbors, config["estimator"], self.cost_weight,
            prior=self.prior(query_id, candidates) if config["estimator"] == "bayesian" else None,
        )
        return estimate["utility"], estimate["support"]

    def score_variant(self, config):
        scores, supports = {}, {}
        for index, key in enumerate(self.ids, 1):
            scores[key], supports[key] = self.query_score(key, config)
            if index % 250 == 0 or index == len(self.ids):
                print(f"scoring={index}/{len(self.ids)}", flush=True)
        return scores, supports


def top_count(ids, scores, count):
    finite = [key for key in ids if math.isfinite(scores[key])]
    if len(finite) < count:
        raise ValueError(f"Only {len(finite)} routable examples for matched count {count}")
    return set(sorted(finite, key=lambda key: (-scores[key], key))[:count])


def fold_analysis(ids, baseline, reference_predictions, variant_predictions):
    rows = []
    for fold in sorted({baseline[key]["outer_fold"] for key in ids}):
        fold_ids = [key for key in ids if baseline[key]["outer_fold"] == fold]
        gold = [baseline[key]["gold_label"] for key in fold_ids]
        reference = [reference_predictions[ids.index(key)] for key in fold_ids]
        candidate = [variant_predictions[ids.index(key)] for key in fold_ids]
        from errorflow.evaluation_v2 import accuracy, macro_f1
        rows.append({
            "outer_fold": fold, "n": len(fold_ids),
            "delta_accuracy_reference_minus_variant": accuracy(reference, gold) - accuracy(candidate, gold),
            "delta_macro_f1_reference_minus_variant": macro_f1(reference, gold) - macro_f1(candidate, gold),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--groups", nargs="+", choices=sorted(GROUPS), default=sorted(GROUPS))
    parser.add_argument("--sbert-cache", type=Path)
    parser.add_argument("--e5-cache", type=Path)
    parser.add_argument("--action", default="conflicting_rescue_v1")
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--utility-threshold", type=float, default=0.0)
    parser.add_argument("--group-threshold", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    baseline_rows = list(read_jsonl(args.baseline_records))
    memory_rows = list(read_jsonl(args.memory))
    baseline_rows, records, _, grouping = prepare_grouped_artifacts(
        baseline_rows, memory_rows, args.group_threshold, args.folds, args.seed,
    )
    baseline = indexed(baseline_rows, "baseline")
    candidate = indexed(read_jsonl(args.candidate), "candidate")
    examples = indexed(read_jsonl(args.examples), "examples")
    if set(baseline) != set(candidate) or set(baseline) != set(examples):
        raise ValueError("Baseline, candidate, and examples IDs must match exactly")
    ids = sorted(baseline)
    selected_names = [name for group in args.groups for name in GROUPS[group]]
    selected_names = list(dict.fromkeys(["C0_full_r6_reference", *selected_names]))
    if any(VARIANTS[name]["claim_metric"] == "sbert" or VARIANTS[name]["evidence_metric"] == "sbert" for name in selected_names) and not args.sbert_cache:
        raise ValueError("Selected groups require --sbert-cache")
    if any(VARIANTS[name]["claim_metric"] == "e5" for name in selected_names) and not args.e5_cache:
        raise ValueError("Selected groups require --e5-cache")
    engine = Engine(ids, baseline, records, examples, args.action, args.cost_weight, args.sbert_cache, args.e5_cache)
    base_metrics, _ = evaluate_selection(ids, baseline, candidate, set())
    scores_by_variant, supports_by_variant = {}, {}
    for name in selected_names:
        print(f"VARIANT_START {name}", flush=True)
        scores_by_variant[name], supports_by_variant[name] = engine.score_variant(VARIANTS[name])

    reference_scores = scores_by_variant["C0_full_r6_reference"]
    reference_natural = {key for key in ids if reference_scores[key] > args.utility_threshold}
    matched_count = len(reference_natural)
    gold = [baseline[key]["gold_label"] for key in ids]
    results, prediction_rows, route_rows, prediction_cache = {}, [], [], {}
    for name in selected_names:
        scores, supports = scores_by_variant[name], supports_by_variant[name]
        selections = {
            "natural": {key for key in ids if scores[key] > args.utility_threshold},
            "matched_budget": top_count(ids, scores, matched_count),
        }
        for mode, selected in selections.items():
            metrics, predictions = evaluate_selection(ids, baseline, candidate, selected)
            metrics["delta_accuracy_vs_b1"] = metrics["accuracy"] - base_metrics["accuracy"]
            metrics["delta_macro_f1_vs_b1"] = metrics["macro_f1"] - base_metrics["macro_f1"]
            results[f"{name}:{mode}"] = metrics
            prediction_cache[(name, mode)] = predictions
            route_rows.append({"variant": name, "mode": mode, "selected_ids": sorted(selected)})
            for key, prediction in zip(ids, predictions):
                prediction_rows.append({
                    "example_id": key, "variant": name, "mode": mode,
                    "routed": key in selected, "prediction": prediction,
                    "gold_label": baseline[key]["gold_label"],
                    "estimated_utility": scores[key] if math.isfinite(scores[key]) else None,
                    "estimated_support": supports[key],
                    "outer_fold": baseline[key]["outer_fold"],
                })

    reference_predictions = prediction_cache[("C0_full_r6_reference", "matched_budget")]
    comparisons, raw_p = {}, {}
    for name in selected_names:
        if name == "C0_full_r6_reference":
            continue
        predictions = prediction_cache[(name, "matched_budget")]
        test = mcnemar_exact(gold, predictions, reference_predictions)
        raw_p[name] = test["exact_two_sided_p"]
        comparisons[name] = {
            "mcnemar_reference_vs_variant": test,
            "paired_bootstrap_reference_minus_variant": paired_bootstrap(
                gold, predictions, reference_predictions, args.bootstrap_samples, args.seed,
            ),
            "fold_analysis": fold_analysis(ids, baseline, reference_predictions, predictions),
        }
    adjusted = holm_adjust(raw_p)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_accuracy_p"] = value

    summary = {
        "evaluation": "grouped_fold_excluded_group_c_substitution_ablation",
        "configuration": {
            "action": args.action, "cost_weight": args.cost_weight,
            "utility_threshold": args.utility_threshold, "group_threshold": args.group_threshold,
            "folds": args.folds, "seed": args.seed,
            "groups": args.groups, "matched_route_count": matched_count,
            "matched_route_rate": matched_count / len(ids),
            "rrf_constant": 60,
            "soft_label_mismatch_penalty": 0.5,
            "confidence_bin_width": 0.2, "margin_bin_width": 0.1,
            "bayesian_prior_strength": 2.0,
        },
        "grouping": grouping, "baseline": base_metrics,
        "variants": {name: VARIANTS[name] for name in selected_names},
        "results": results, "matched_budget_reference_comparisons": comparisons,
        "excluded_p2_comparators": {
            "C4-4_source_fold_tuned_weights": "requires nested source-inner model selection; run dedicated P2 evaluator",
            "C6-4_local_logistic_model": "trained-router comparator, not part of the training-free family; run dedicated P2 evaluator",
        },
        "protocol_notes": [
            "C4-1 is implementation-equivalent to Group-A A7 under the same grouped protocol.",
            "Dense caches are validated by exact example ordering and an adjacent manifest.",
            "All target examples and their grouped outer folds are excluded from usable memory.",
            "Natural routing uses utility > 0; matched routing uses the C0 natural route count.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(args.output_dir / "routes.jsonl", route_rows)
    print(json.dumps({
        "matched_route_count": matched_count,
        "matched_route_rate": matched_count / len(ids),
        "variants": len(selected_names), "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
