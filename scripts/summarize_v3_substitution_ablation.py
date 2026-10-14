"""Print a compact, copyable summary of Group-C ablation artifacts."""

import argparse
import json
from pathlib import Path


def print_summary(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"\nSUMMARY {path}")
    print("VARIANT\tMODE\tROUTED\tRATE\tHELPFUL\tHARMFUL\tNET\tACC\tMACRO_F1\tHOLM_ACC_P")
    comparisons = report.get("matched_budget_reference_comparisons", report.get("reference_comparisons", {}))
    for key, row in report["results"].items():
        variant, mode = key.rsplit(":", 1)
        comparison = comparisons.get(variant, {}) if mode == "matched_budget" else {}
        print(
            variant, mode, row["routed"], f'{row["route_rate"]:.6f}',
            row["helpful"], row["harmful"], row["net"],
            f'{row["accuracy"]:.6f}', f'{row["macro_f1"]:.6f}',
            comparison.get("holm_adjusted_accuracy_p", "NA"), sep="\t",
        )
    print("\nMATCHED-BUDGET SIGNIFICANCE")
    for variant, row in comparisons.items():
        mc = row["mcnemar_reference_vs_variant"]
        boot = row["paired_bootstrap_reference_minus_variant"]
        print(variant, json.dumps({
            "mcnemar_exact_p": mc["exact_two_sided_p"],
            "holm_accuracy_p": row.get("holm_adjusted_accuracy_p"),
            "accuracy_delta_ci95": boot["accuracy_delta_ci95"],
            "macro_f1_delta_ci95": boot["macro_f1_delta_ci95"],
        }, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.summaries:
        print_summary(path)


if __name__ == "__main__":
    main()
