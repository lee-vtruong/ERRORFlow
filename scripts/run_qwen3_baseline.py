"""Run Qwen3 baseline over a small JSONL input file."""

import argparse
import json
from pathlib import Path

from errorflow.verifier import Qwen3TransformersVerifier, VerifierConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="unknown")
    parser.add_argument("--intervention", default="none", choices=["none", "evidence_critic", "conflict_check"])
    args = parser.parse_args()
    verifier = Qwen3TransformersVerifier(VerifierConfig(args.model, checkpoint="base"))
    instructions = {
        "none": "",
        "evidence_critic": "Before choosing a label, inspect each evidence sentence and explain whether it directly supports, contradicts, or fails to establish the claim.",
        "conflict_check": "Before choosing a label, explicitly check for both supporting and contradicting evidence. Choose NOT ENOUGH INFO when neither direction is established.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            record = verifier.predict(row["claim_id"], row["claim"], row.get("evidence", []), split=args.split, instruction=instructions[args.intervention])
            record.metadata["intervention"] = args.intervention
            target.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            print(record.claim_id, record.prediction)


if __name__ == "__main__":
    main()
