"""Run frozen Qwen over canonical V2 JSONL with safe resume."""

import argparse
import json
from pathlib import Path

from errorflow.io import read_jsonl
from errorflow.verifier import Qwen3TransformersVerifier, VerifierConfig

LABEL_SPACES = {
    "fever": ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO"),
    "averitec": ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO", "CONFLICTING"),
}

INSTRUCTION_PROFILES = {
    "none": "",
    "averitec_v1": (
        "Use these definitions strictly. SUPPORTED: the evidence directly establishes the claim. "
        "REFUTED: the evidence directly establishes that the claim is false. "
        "NOT ENOUGH INFO: the evidence is insufficient to establish either support or refutation. "
        "CONFLICTING: credible evidence supports and contradicts the claim, or the claim cherry-picks a mixed record. "
        "Do not treat merely incomplete evidence as CONFLICTING."
    ),
}


def evidence_texts(row: dict) -> list[str]:
    return [item["text"] if isinstance(item, dict) else str(item) for item in row.get("evidence", [])]


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = [str(row["example_id"]) for row in read_jsonl(path)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate example_id in resume output: {path}")
    return set(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label-set", choices=sorted(LABEL_SPACES), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--decoding", choices=["generate", "label-score"], default="generate")
    parser.add_argument("--instruction-profile", choices=sorted(INSTRUCTION_PROFILES), default="none")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = list(read_jsonl(args.input))
    input_ids = [str(row["example_id"]) for row in rows]
    if len(input_ids) != len(set(input_ids)):
        raise ValueError(f"Duplicate example_id in input: {args.input}")
    completed = existing_ids(args.output) if args.resume else set()
    unknown = completed - set(input_ids)
    if unknown:
        raise ValueError(f"Resume output has {len(unknown)} unknown IDs")
    pending = [row for row in rows if str(row["example_id"]) not in completed]
    verifier = Qwen3TransformersVerifier(VerifierConfig(
        args.model, max_new_tokens=args.max_new_tokens, label_space=LABEL_SPACES[args.label_set]
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and args.output.exists() else "w"
    print(f"V2_INFERENCE_START total={len(rows)} completed={len(completed)} remaining={len(pending)} labels={LABEL_SPACES[args.label_set]}", flush=True)
    with args.output.open(mode, encoding="utf-8") as target:
        for offset, row in enumerate(pending, 1):
            kwargs = {
                "claim_id": str(row["example_id"]), "claim": row["claim"],
                "evidence": evidence_texts(row), "split": args.split,
                "instruction": INSTRUCTION_PROFILES[args.instruction_profile],
            }
            record = verifier.predict(**kwargs) if args.decoding == "generate" else verifier.score_labels(**kwargs)
            payload = record.to_dict()
            payload["example_id"] = payload.pop("claim_id")
            payload["dataset"] = row.get("dataset")
            payload["metadata"].update({
                "instruction_profile": args.instruction_profile,
                "decoding_mode": args.decoding,
                "configured_max_new_tokens": args.max_new_tokens,
            })
            target.write(json.dumps(payload, ensure_ascii=False) + "\n")
            target.flush()
            print(f"progress={len(completed)+offset}/{len(rows)} example_id={payload['example_id']} prediction={payload['prediction']}", flush=True)
    print(f"V2_INFERENCE_DONE total={len(rows)} session_rows={len(pending)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
