"""Audit SciFact label projection and controlled V3 prediction variants."""

import argparse
import hashlib
import json
from pathlib import Path

from errorflow.evaluation_v2 import accuracy, macro_f1
from errorflow.io import atomic_write_json, read_jsonl, write_jsonl
from errorflow.v3_statistics import mcnemar_exact, paired_bootstrap


PROJECTION = {"CONFLICTING": "NOT ENOUGH INFO"}


def indexed(rows, name):
    result = {}
    for row in rows:
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in {name}: {key}")
        result[key] = row
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project(label):
    return PROJECTION.get(label, label)


def per_class(gold, predicted):
    report = {}
    for label in sorted(set(gold) | set(predicted)):
        tp = sum(g == p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        report[label] = {
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": sum(g == label for g in gold),
        }
    return report


def metrics(gold, predicted, passes):
    return {
        "n": len(gold), "accuracy": accuracy(predicted, gold),
        "macro_f1": macro_f1(predicted, gold),
        "per_class": per_class(gold, predicted),
        "avg_model_forward_passes": sum(passes) / len(passes),
    }


def comparison(gold, before, after, samples, seed):
    return {
        "mcnemar_exact": mcnemar_exact(gold, before, after),
        "paired_bootstrap": paired_bootstrap(gold, before, after, samples, seed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--b1-raw", required=True, type=Path)
    parser.add_argument("--a3-raw", required=True, type=Path)
    parser.add_argument("--r6-raw", required=True, type=Path)
    parser.add_argument("--b1-direct3", type=Path)
    parser.add_argument("--a3-direct3", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    gold_rows = indexed(read_jsonl(args.gold), "gold")
    b1 = indexed(read_jsonl(args.b1_raw), "b1_raw")
    a3 = indexed(read_jsonl(args.a3_raw), "a3_raw")
    r6 = indexed(read_jsonl(args.r6_raw), "r6_raw")
    ids = sorted(gold_rows)
    for name, rows in (("b1_raw", b1), ("a3_raw", a3), ("r6_raw", r6)):
        if set(rows) != set(ids):
            raise ValueError(f"{name} IDs do not exactly match gold")
    direct = {}
    for name, path in (("b1_direct3", args.b1_direct3), ("a3_direct3", args.a3_direct3)):
        if path:
            direct[name] = indexed(read_jsonl(path), name)
            if set(direct[name]) != set(ids):
                raise ValueError(f"{name} IDs do not exactly match gold")

    expected = [gold_rows[key]["label"] for key in ids]
    b1_raw = [b1[key]["prediction"] for key in ids]
    a3_raw = [a3[key]["prediction"] for key in ids]
    r6_raw = [r6[key]["prediction"] for key in ids]
    routed = [bool(r6[key].get("routed")) for key in ids]
    b1_projected = [project(label) for label in b1_raw]
    a3_projected = [project(label) for label in a3_raw]
    r6_projected = [project(label) for label in r6_raw]
    r6_passes = [float(r6[key].get("metadata", {}).get("model_forward_passes", 4 + 4 * int(flag)))
                 for key, flag in zip(ids, routed)]
    systems = {
        "b1_raw_four_label": metrics(expected, b1_raw, [4.0] * len(ids)),
        "a3_all_raw_four_label": metrics(expected, a3_raw, [8.0] * len(ids)),
        "r6_memory_no_a3_raw": metrics(expected, b1_raw, [4.0] * len(ids)),
        "r6_full_raw_four_label": metrics(expected, r6_raw, r6_passes),
        "b1_projected_three_label": metrics(expected, b1_projected, [4.0] * len(ids)),
        "a3_all_projected_three_label": metrics(expected, a3_projected, [8.0] * len(ids)),
        "r6_memory_no_a3_projected": metrics(expected, b1_projected, [4.0] * len(ids)),
        "r6_full_projected_three_label": metrics(expected, r6_projected, r6_passes),
    }
    if direct:
        for name, rows in direct.items():
            predictions = [rows[key]["prediction"] for key in ids]
            component_passes = [
                float(rows[key].get("metadata", {}).get("model_forward_passes", 3)) for key in ids
            ]
            passes = ([2.0 * value for value in component_passes]
                      if name == "a3_direct3" else component_passes)
            systems[name] = metrics(expected, predictions, passes)

    transitions = []
    for position, key in enumerate(ids):
        row = {
            "example_id": key, "gold_label": expected[position],
            "b1_raw_label": b1_raw[position], "r6_raw_label": r6_raw[position],
            "a3_raw_label": a3_raw[position],
            "b1_projected_label": b1_projected[position],
            "r6_projected_label": r6_projected[position],
            "a3_projected_label": a3_projected[position],
            "was_routed": routed[position],
            "raw_verdict_changed": b1_raw[position] != r6_raw[position],
            "b1_projection_changed_label": b1_raw[position] != b1_projected[position],
            "r6_projection_changed_label": r6_raw[position] != r6_projected[position],
            "a3_projection_changed_label": a3_raw[position] != a3_projected[position],
            "genuine_raw_correction": b1_raw[position] != expected[position] and r6_raw[position] == expected[position],
            "projected_correction": b1_projected[position] != expected[position] and r6_projected[position] == expected[position],
            "b1_projection_only_gain": b1_raw[position] != expected[position] and b1_projected[position] == expected[position],
            "r6_projection_only_gain": r6_raw[position] != expected[position] and r6_projected[position] == expected[position],
            "a3_projection_only_gain": a3_raw[position] != expected[position] and a3_projected[position] == expected[position],
            "projection_hides_r6_four_label_error": r6_raw[position] != expected[position] and r6_projected[position] == expected[position],
        }
        if (row["raw_verdict_changed"] or row["b1_projection_changed_label"]
                or row["r6_projection_changed_label"] or row["a3_projection_changed_label"]):
            transitions.append(row)

    projection_counts = {
        "b1_changed": sum(a != b for a, b in zip(b1_raw, b1_projected)),
        "b1_correctness_gained": sum(a != g and b == g for g, a, b in zip(expected, b1_raw, b1_projected)),
        "b1_correctness_lost": sum(a == g and b != g for g, a, b in zip(expected, b1_raw, b1_projected)),
        "r6_changed": sum(a != b for a, b in zip(r6_raw, r6_projected)),
        "r6_correctness_gained": sum(a != g and b == g for g, a, b in zip(expected, r6_raw, r6_projected)),
        "r6_correctness_lost": sum(a == g and b != g for g, a, b in zip(expected, r6_raw, r6_projected)),
        "a3_changed": sum(a != b for a, b in zip(a3_raw, a3_projected)),
        "a3_correctness_gained": sum(a != g and b == g for g, a, b in zip(expected, a3_raw, a3_projected)),
        "a3_correctness_lost": sum(a == g and b != g for g, a, b in zip(expected, a3_raw, a3_projected)),
    }
    summary = {
        "evaluation": "scifact_label_adapter_audit",
        "n": len(ids), "projection": PROJECTION,
        "adapter_timing": {
            "routing_feature_space": "raw AVeriTeC four-label B1 scores",
            "utility_space": "raw AVeriTeC four-label source-memory outcomes",
            "projection_stage": "final predictions only, after routing and A3 selection",
            "same_mapping_applied_to_b1_and_r6": True,
        },
        "controls": {
            "projection_only": "Compare each raw prediction stream with its own projected stream.",
            "memory_without_a3": "Reuse R6 route decisions but retain B1 predictions; verdict metrics equal B1 by construction.",
            "a3_without_memory_selection": "Always-on A3 over all 300 examples.",
            "direct_three_label": (
                "Separate ontology-specific scoring sensitivity; not mathematically equivalent to projection and not routed "
                "through four-label AVeriTeC memory."
            ),
        },
        "systems": systems,
        "route_accounting": {"routed": sum(routed), "route_rate": sum(routed) / len(routed)},
        "projection_accounting": projection_counts,
        "comparisons": {
            "r6_vs_b1_raw": comparison(expected, b1_raw, r6_raw, args.samples, args.seed),
            "r6_vs_b1_projected": comparison(expected, b1_projected, r6_projected, args.samples, args.seed),
            "a3_all_vs_b1_raw": comparison(expected, b1_raw, a3_raw, args.samples, args.seed),
            "a3_all_vs_b1_projected": comparison(expected, b1_projected, a3_projected, args.samples, args.seed),
            "r6_vs_a3_all_projected": comparison(expected, a3_projected, r6_projected, args.samples, args.seed),
            "b1_projection_only": comparison(expected, b1_raw, b1_projected, args.samples, args.seed),
            "r6_projection_only": comparison(expected, r6_raw, r6_projected, args.samples, args.seed),
            "a3_projection_only": comparison(expected, a3_raw, a3_projected, args.samples, args.seed),
        },
        "transition_counts": {
            "rows_written": len(transitions),
            "raw_genuine_corrections": sum(row["genuine_raw_correction"] for row in transitions),
            "projected_corrections": sum(row["projected_correction"] for row in transitions),
            "r6_projection_only_gains": sum(row["r6_projection_only_gain"] for row in transitions),
            "a3_projection_only_gains": sum(row["a3_projection_only_gain"] for row in transitions),
        },
        "input_artifacts": {
            "gold": {"path": str(args.gold), "sha256": sha256_file(args.gold)},
            "b1_raw": {"path": str(args.b1_raw), "sha256": sha256_file(args.b1_raw)},
            "a3_raw": {"path": str(args.a3_raw), "sha256": sha256_file(args.a3_raw)},
            "r6_raw": {"path": str(args.r6_raw), "sha256": sha256_file(args.r6_raw)},
        },
        "external_validation_warning": (
            "SciFact dev labels are used only for this audit/evaluation. Any adapter changed after reading these results "
            "would require a new untouched target split for confirmatory validation."
        ),
    }
    for name, path in (("b1_direct3", args.b1_direct3), ("a3_direct3", args.a3_direct3)):
        if path:
            summary["input_artifacts"][name] = {"path": str(path), "sha256": sha256_file(path)}
    if "b1_direct3" in direct:
        b1_direct = [direct["b1_direct3"][key]["prediction"] for key in ids]
        summary["comparisons"]["b1_direct3_vs_b1_projected"] = comparison(
            expected, b1_projected, b1_direct, args.samples, args.seed,
        )
    if "a3_direct3" in direct:
        a3_direct = [direct["a3_direct3"][key]["prediction"] for key in ids]
        summary["comparisons"]["a3_direct3_vs_a3_projected"] = comparison(
            expected, a3_projected, a3_direct, args.samples, args.seed,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "transitions.jsonl", transitions)
    print(json.dumps({
        "systems": systems, "route_accounting": summary["route_accounting"],
        "projection_accounting": projection_counts,
        "transition_counts": summary["transition_counts"],
    }, indent=2))
    print(f"SCIFACT_ADAPTER_AUDIT_DONE rows={len(ids)} transitions={len(transitions)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
