"""Join official SciFact claims to gold rationale sentences for oracle evaluation."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl, write_jsonl


LABELS = {"SUPPORT": "SUPPORTED", "CONTRADICT": "REFUTED"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(corpus_rows, claim_rows, split):
    corpus = {str(row["doc_id"]): row for row in corpus_rows}
    output, missing_docs, invalid_sentences = [], 0, 0
    for claim in claim_rows:
        evidence_map = claim.get("evidence") or {}
        raw_labels, evidence = set(), []
        for doc_id, rationale_sets in evidence_map.items():
            document = corpus.get(str(doc_id))
            if document is None:
                missing_docs += 1
                continue
            abstract = document.get("abstract", [])
            for group_index, rationale in enumerate(rationale_sets):
                raw_label = str(rationale["label"]).upper()
                if raw_label not in LABELS:
                    raise ValueError(f"Unknown SciFact rationale label: {raw_label}")
                raw_labels.add(raw_label)
                sentence_ids = [int(index) for index in rationale.get("sentences", [])]
                valid_ids = [index for index in sentence_ids if 0 <= index < len(abstract)]
                invalid_sentences += len(sentence_ids) - len(valid_ids)
                text = " ".join(str(abstract[index]) for index in valid_ids)
                if text:
                    evidence.append({
                        "evidence_id": f"{doc_id}:{group_index}", "text": text,
                        "title": document.get("title"), "url": None,
                        "source": "scifact_gold_rationale",
                        "metadata": {"doc_id": str(doc_id), "sentence_ids": valid_ids},
                    })
        if not raw_labels:
            label = "NOT ENOUGH INFO"
        elif len(raw_labels) == 1:
            label = LABELS[next(iter(raw_labels))]
        else:
            raise ValueError(f"Mixed SciFact gold labels for claim {claim['id']}: {sorted(raw_labels)}")
        output.append({
            "example_id": f"scifact-{split}:{claim['id']}", "dataset": "SciFact",
            "split": split, "claim": claim["claim"], "label": label,
            "evidence": evidence,
            "metadata": {"cited_doc_ids": claim.get("cited_doc_ids", []),
                         "evidence_protocol": "official_gold_rationale_oracle"},
        })
    return output, {"missing_documents": missing_docs, "invalid_sentence_indices": invalid_sentences}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    claims = list(read_jsonl(args.claims))
    if args.limit is not None:
        claims = claims[:args.limit]
    rows, diagnostics = prepare(read_jsonl(args.corpus), claims, args.split)
    manifest = {
        "dataset": "SciFact", "split": args.split,
        "evidence_protocol": "official_gold_rationale_oracle",
        "rows": len(rows), "labels": dict(Counter(row["label"] for row in rows)),
        "corpus_sha256": sha256_file(args.corpus), "claims_sha256": sha256_file(args.claims),
        **diagnostics,
    }
    write_jsonl(args.output, rows)
    atomic_write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))
    print(f"SCIFACT_ORACLE_READY rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
