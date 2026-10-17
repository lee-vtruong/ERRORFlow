"""Print compact, pasteable results for decisive Standard-pair experiments."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.summaries:
        report = json.loads(path.read_text(encoding="utf-8"))
        print(f"SUMMARY {path}")
        print(f"DATASET {report['dataset']} N={report['n']}")
        print("METHOD\tROUTED\tRATE\tHELPFUL\tHARMFUL\tNET\tACC\tMACRO_F1\tPASSES")
        for name, row in report["results"].items():
            print(
                name, row["routed"], f"{row['route_rate']:.6f}", row["helpful"],
                row["harmful"], row["net"], f"{row['accuracy']:.6f}",
                f"{row['macro_f1']:.6f}", f"{row['avg_model_forward_passes']:.3f}", sep="\t",
            )
        print("HYPOTHESIS", json.dumps(report["hypothesis_test"], sort_keys=True))
        for name, comparison in report["comparisons"].items():
            grouped = comparison["grouped_paired_bootstrap"]
            print(name, json.dumps({
                "mcnemar_p": comparison["mcnemar_exact"]["exact_two_sided_p"],
                "accuracy_ci95": grouped["accuracy_delta_ci95"],
                "macro_f1_ci95": grouped["macro_f1_delta_ci95"],
            }, sort_keys=True))
        print("PROVENANCE", json.dumps(report["standard_pair_provenance"], sort_keys=True))


if __name__ == "__main__":
    main()
