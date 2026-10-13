"""Bounded, hash-validated Qwen action runner for ERRORFlow V3."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from errorflow.io import atomic_write_json, canonical_json, config_hash, read_jsonl
from errorflow.v3_prompts import (
    B1_CONFLICT_AUDIT,
    SCIFACT_THREE_LABEL_PROMPTS,
    V3_ACTION_PROMPTS,
    assert_b1_immutable,
    prompt_sha256,
)
from errorflow.verifier import Qwen3TransformersVerifier, VerifierConfig


LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO", "CONFLICTING")
SCIFACT_LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO")
LABEL_PROTOCOLS = {"averitec4": LABELS, "scifact3": SCIFACT_LABELS}
ACTION_PROMPTS = {"conflict_audit_v1": B1_CONFLICT_AUDIT, **V3_ACTION_PROMPTS}


def resolve_protocol(action, protocol):
    labels = LABEL_PROTOCOLS[protocol]
    if protocol == "scifact3":
        if action not in SCIFACT_THREE_LABEL_PROMPTS:
            raise ValueError(f"No frozen SciFact three-label prompt for action: {action}")
        return labels, SCIFACT_THREE_LABEL_PROMPTS[action]
    return labels, ACTION_PROMPTS[action]


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_texts(row):
    return [item.get("text", "") if isinstance(item, dict) else str(item) for item in row.get("evidence", [])]


def sample_cache_key(row, model_hash, prompt_hash, generation_hash):
    evidence_payload = [{
        "evidence_id": item.get("evidence_id"), "text": item.get("text", ""),
        "title": item.get("title"), "url": item.get("url"), "source": item.get("source"),
    } if isinstance(item, dict) else {"text": str(item)} for item in row.get("evidence", [])]
    evidence_hash = sha256_text(canonical_json(evidence_payload))
    payload = {
        "sample_id": str(row["example_id"]), "model_hash": model_hash,
        "prompt_hash": prompt_hash, "evidence_hash": evidence_hash,
        "generation_config_hash": generation_hash,
    }
    return config_hash(payload, length=32), evidence_hash


def existing_rows(path):
    if not path.exists():
        return {}
    result = {}
    for row in read_jsonl(path):
        key = str(row["example_id"])
        if key in result:
            raise ValueError(f"Duplicate example_id in resume output: {key}")
        result[key] = row
    return result


def validate_cached_rows(cached, inputs, model_hash, prompt_hash, generation_hash):
    by_id = {str(row["example_id"]): row for row in inputs}
    unknown = set(cached) - set(by_id)
    if unknown:
        raise ValueError(f"Resume output contains {len(unknown)} unknown IDs")
    for key, row in cached.items():
        expected, _ = sample_cache_key(by_id[key], model_hash, prompt_hash, generation_hash)
        actual = row.get("metadata", {}).get("cache_key")
        if actual != expected:
            raise ValueError(f"Cache key mismatch for {key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--action", required=True, choices=sorted(ACTION_PROMPTS))
    parser.add_argument("--label-protocol", choices=sorted(LABEL_PROTOCOLS), default="averitec4")
    parser.add_argument("--split", default="train")
    destinations = parser.add_mutually_exclusive_group(required=True)
    destinations.add_argument("--output", type=Path)
    destinations.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-llm-calls", type=int)
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--external-evaluation", action="store_true")
    args = parser.parse_args()
    assert_b1_immutable()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.max_llm_calls is not None and args.max_llm_calls < 1:
        raise ValueError("--max-llm-calls must be positive")
    output = args.output or args.output_dir / "predictions.jsonl"
    output_dir = output.parent
    manifest_path = output_dir / (output.stem + ".manifest.json")
    rows = list(read_jsonl(args.input))
    if args.max_samples is not None:
        rows = rows[:args.max_samples]
    ids = [str(row["example_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Input contains duplicate example IDs")
    datasets = {row.get("dataset") for row in rows}
    if args.external_evaluation:
        if not args.split.startswith("external_"):
            raise ValueError("External evaluation split must start with external_")
        if not datasets or not datasets <= {"SciFact", "Climate-FEVER"}:
            raise ValueError("External evaluation accepts only SciFact or Climate-FEVER")
    elif args.split != "train" or any(key.startswith("dev:") for key in ids):
        raise ValueError("V3 development action runner accepts train split only")

    if args.label_protocol == "scifact3" and (not args.external_evaluation or datasets != {"SciFact"}):
        raise ValueError("scifact3 requires --external-evaluation on SciFact only")
    labels, prompt = resolve_protocol(args.action, args.label_protocol)
    prompt_hash = prompt_sha256(prompt)
    model_hash = sha256_text(args.model)
    # Preserve the historical four-label generation hash/cache contract.
    generation = {"decoding": "mean_label_log_probability", "labels": labels, "forward_passes": len(labels)}
    if args.label_protocol != "averitec4":
        generation["label_protocol"] = args.label_protocol
    generation_hash = config_hash(generation, length=32)
    run_config = {
        "model": args.model, "model_hash_type": "sha256_model_identifier", "model_hash": model_hash,
        "input": str(args.input), "input_sha256": sha256_file(args.input),
        "action": args.action, "prompt_sha256": prompt_hash, "split": args.split,
        "generation": generation, "generation_config_hash": generation_hash,
        "max_samples": args.max_samples, "seed": args.seed,
    }
    if args.external_evaluation:
        run_config["external_evaluation"] = True
    manifest = {
        "schema_version": 3, "config_id": config_hash(run_config, length=32), "config": run_config,
        "status": "dry_run" if args.dry_run else "running", "output": str(output),
        "requested_rows": len(rows), "completed_rows": 0,
        "llm_request_definition": (
            f"one score_labels call per example; {len(labels)} model forward passes per request"
        ),
    }
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("config_id") != manifest["config_id"]:
            raise ValueError("Existing manifest config hash does not match this run")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    cached = existing_rows(output) if args.resume else {}
    if output.exists() and not args.resume:
        raise ValueError(f"Output already exists; use --resume only for the same manifest: {output}")
    validate_cached_rows(cached, rows, model_hash, prompt_hash, generation_hash)
    pending = [row for row in rows if str(row["example_id"]) not in cached]
    if args.max_llm_calls is not None:
        pending = pending[:args.max_llm_calls]
    manifest["completed_rows"] = len(cached)
    atomic_write_json(manifest_path, manifest)
    verifier = Qwen3TransformersVerifier(VerifierConfig(args.model, seed=args.seed, label_space=labels))
    started = time.perf_counter()
    mode = "a" if args.resume and output.exists() else "w"
    session_rows = 0
    with output.open(mode, encoding="utf-8", newline="\n") as target:
        for row in pending:
            if args.max_runtime_seconds is not None and time.perf_counter() - started >= args.max_runtime_seconds:
                break
            cache_key, evidence_hash = sample_cache_key(row, model_hash, prompt_hash, generation_hash)
            record = verifier.score_labels(
                str(row["example_id"]), row["claim"], evidence_texts(row),
                split=args.split, instruction=prompt,
            ).to_dict()
            record["example_id"] = record.pop("claim_id")
            record["dataset"] = row.get("dataset")
            record["metadata"].update({
                "instruction_profile": args.action, "decoding_mode": "label-score",
                "prompt_sha256": prompt_hash, "model_hash": model_hash,
                "evidence_hash": evidence_hash, "generation_config_hash": generation_hash,
                "cache_key": cache_key, "gold_label_accessed": False,
                "label_protocol": args.label_protocol,
                "model_forward_passes": len(labels), "llm_requests": 1,
            })
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()
            session_rows += 1
            print(f"progress={len(cached)+session_rows}/{len(rows)} example_id={record['example_id']} prediction={record['prediction']}", flush=True)
    completed = len(existing_rows(output))
    manifest.update({
        "status": "complete" if completed == len(rows) else "partial",
        "completed_rows": completed, "session_rows": session_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "model_forward_passes": completed * len(labels),
    })
    atomic_write_json(manifest_path, manifest)
    print(f"V3_ACTION_DONE status={manifest['status']} completed={completed}/{len(rows)} output={output}")


if __name__ == "__main__":
    main()
