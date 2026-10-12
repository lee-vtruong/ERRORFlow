"""Rerank retrieval candidates with an off-the-shelf cross-encoder."""
import argparse
import json
import time
from pathlib import Path


def rerank_row(row, scores, model_name):
    evidence = row.get("evidence", [])
    metadata = row.get("evidence_metadata", [])
    if len(evidence) != len(metadata) or len(evidence) != len(scores):
        raise ValueError(f"unaligned evidence, metadata, or scores for claim {row.get('claim_id')}")
    ranked = sorted(
        zip(evidence, metadata, scores, range(1, len(scores) + 1)),
        key=lambda item: (-float(item[2]), item[3]),
    )
    output = dict(row)
    output["evidence"] = [item[0] for item in ranked]
    output["evidence_metadata"] = []
    for rank, (_, source_metadata, score, old_rank) in enumerate(ranked, start=1):
        updated = dict(source_metadata)
        updated["pre_rerank_rank"] = old_rank
        updated["rank"] = rank
        updated["reranker_score"] = float(score)
        output["evidence_metadata"].append(updated)
    output["retrieval"] = {
        "method": "cross_encoder_rerank",
        "candidate_retrieval": row.get("retrieval", {}),
        "reranker_model": model_name,
        "k": len(ranked),
        "provenance": True,
    }
    return output


def existing_ids(path):
    completed = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                claim_id = str(json.loads(line)["claim_id"])
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"invalid resume output at line {line_number}: {error}") from error
            if claim_id in completed:
                raise ValueError(f"duplicate claim_id in resume output: {claim_id}")
            completed.add(claim_id)
    return completed


def score_pairs(model, tokenizer, claim, passages, batch_size, max_length, device, torch):
    scores = []
    for start in range(0, len(passages), batch_size):
        batch = passages[start:start + batch_size]
        inputs = tokenizer(
            [claim] * len(batch),
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs, return_dict=True).logits
        if logits.ndim == 2 and logits.shape[1] == 1:
            batch_scores = logits[:, 0]
        elif logits.ndim == 2:
            batch_scores = torch.softmax(logits.float(), dim=-1)[:, -1]
        else:
            batch_scores = logits.reshape(-1)
        scores.extend(batch_scores.float().cpu().tolist())
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    with args.input.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    input_ids = {str(row["claim_id"]) for row in rows}
    completed = existing_ids(args.output) if args.resume else set()
    unknown = completed - input_ids
    if unknown:
        raise ValueError(f"resume output contains {len(unknown)} IDs absent from input")
    pending = [row for row in rows if str(row["claim_id"]) not in completed]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.type == "cuda" else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, dtype=dtype, local_files_only=args.local_files_only
    ).to(device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.resume and completed else "w"
    started = time.perf_counter()
    print(
        f"RERANK_START total={len(rows)} completed={len(completed)} remaining={len(pending)} "
        f"model={args.model} batch_size={args.batch_size} max_length={args.max_length}",
        flush=True,
    )
    with args.output.open(output_mode, encoding="utf-8") as target:
        for done, row in enumerate(pending, start=1):
            scores = score_pairs(
                model, tokenizer, row["claim"], row.get("evidence", []),
                args.batch_size, args.max_length, device, torch
            )
            target.write(json.dumps(rerank_row(row, scores, args.model), ensure_ascii=False) + "\n")
            target.flush()
            if done == 1 or done % args.progress_every == 0 or done == len(pending):
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0.0
                eta = (len(pending) - done) / rate if rate else 0.0
                print(
                    f"progress={len(completed)+done}/{len(rows)} session_done={done}/{len(pending)} "
                    f"rate={rate:.2f}_claims/s eta={eta:.1f}s",
                    flush=True,
                )
    print(
        f"RERANK_DONE total={len(rows)} resumed_from={len(completed)} session_rows={len(pending)} "
        f"elapsed_s={time.perf_counter()-started:.1f} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
