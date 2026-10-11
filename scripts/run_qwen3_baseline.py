"""Run Qwen3 baseline over a small JSONL input file."""

import argparse
import json
from pathlib import Path

from errorflow.verifier import Qwen3TransformersVerifier, VerifierConfig


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
    return rows


def completed_claim_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = load_jsonl(path)
    ids = [str(row["claim_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate claim_id found in resume output: {path}")
    return set(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="unknown")
    parser.add_argument("--intervention", default="none", choices=["none", "evidence_critic", "conflict_check"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    instructions = {
        "none": "",
        "evidence_critic": "Before choosing a label, inspect each evidence sentence and explain whether it directly supports, contradicts, or fails to establish the claim.",
        "conflict_check": "Before choosing a label, explicitly check for both supporting and contradicting evidence. Choose NOT ENOUGH INFO when neither direction is established.",
    }
    rows = load_jsonl(args.input)
    completed = completed_claim_ids(args.output) if args.resume else set()
    input_ids = [str(row["claim_id"]) for row in rows]
    if len(input_ids) != len(set(input_ids)):
        raise ValueError(f"Duplicate claim_id found in input: {args.input}")
    unknown = completed - set(input_ids)
    if unknown:
        raise ValueError(f"Resume output contains {len(unknown)} claim IDs absent from input")
    pending = [row for row in rows if str(row["claim_id"]) not in completed]
    print(
        f"INFERENCE_START total={len(rows)} completed={len(completed)} "
        f"remaining={len(pending)} intervention={args.intervention}",
        flush=True,
    )
    verifier = Qwen3TransformersVerifier(VerifierConfig(args.model, checkpoint="base"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(output_mode, encoding="utf-8") as target:
        for done, row in enumerate(pending, 1):
            record = verifier.predict(row["claim_id"], row["claim"], row.get("evidence", []), split=args.split, instruction=instructions[args.intervention])
            record.metadata["intervention"] = args.intervention
            target.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            target.flush()
            print(
                f"progress={len(completed) + done}/{len(rows)} "
                f"claim_id={record.claim_id} prediction={record.prediction}",
                flush=True,
            )
    print(
        f"INFERENCE_DONE total={len(rows)} resumed_from={len(completed)} "
        f"session_rows={len(pending)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
