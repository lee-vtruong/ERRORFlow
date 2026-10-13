"""Build a deterministic open-retrieval SciFact verifier artifact."""

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

from errorflow.io import atomic_write_json, read_jsonl, write_jsonl


LABELS = {"SUPPORT": "SUPPORTED", "CONTRADICT": "REFUTED"}
TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokens(text):
    return TOKEN.findall(str(text).lower())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BM25:
    def __init__(self, documents, k1=1.2, b=0.75):
        self.documents = [tokens(document) for document in documents]
        self.k1, self.b = k1, b
        self.lengths = [len(document) for document in self.documents]
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.frequencies = [Counter(document) for document in self.documents]
        document_frequency = Counter()
        for document in self.documents:
            document_frequency.update(set(document))
        size = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score(self, query, index):
        return self.score_terms(set(tokens(query)), index)

    def score_terms(self, query_terms, index):
        if not self.avg_length:
            return 0.0
        frequencies = self.frequencies[index]
        length = self.lengths[index]
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (1.0 - self.b + self.b * length / self.avg_length)
            score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / denominator
        return score

    def rank(self, query, count, indices=None):
        candidates = range(len(self.documents)) if indices is None else indices
        query_terms = set(tokens(query))
        return sorted(
            ((index, self.score_terms(query_terms, index)) for index in candidates),
            key=lambda item: (-item[1], item[0]),
        )[:count]


def claim_label(claim):
    labels = {
        str(rationale["label"]).upper()
        for rationale_sets in (claim.get("evidence") or {}).values()
        for rationale in rationale_sets
    }
    if not labels:
        return "NOT ENOUGH INFO"
    if len(labels) != 1 or next(iter(labels)) not in LABELS:
        raise ValueError(f"Unexpected SciFact labels for claim {claim['id']}: {sorted(labels)}")
    return LABELS[next(iter(labels))]


def retrieve(corpus_rows, claim_rows, split, top_documents=5, top_sentences=5, progress_every=0):
    documents = []
    sentences = []
    document_to_sentences = {}
    for document_index, row in enumerate(corpus_rows):
        doc_id = str(row["doc_id"])
        abstract = [str(sentence) for sentence in row.get("abstract", [])]
        documents.append({"doc_id": doc_id, "title": row.get("title"), "abstract": abstract})
        sentence_indices = []
        for sentence_id, text in enumerate(abstract):
            sentence_indices.append(len(sentences))
            sentences.append({
                "doc_id": doc_id, "sentence_id": sentence_id,
                "title": row.get("title"), "text": text,
            })
        document_to_sentences[document_index] = sentence_indices

    document_bm25 = BM25([
        f"{document['title'] or ''} {' '.join(document['abstract'])}" for document in documents
    ])
    sentence_bm25 = BM25([sentence["text"] for sentence in sentences])
    output = []
    started = time.perf_counter()
    for claim_index, claim in enumerate(claim_rows, 1):
        query = str(claim["claim"])
        ranked_documents = document_bm25.rank(query, top_documents)
        candidate_sentences = [
            sentence_index
            for document_index, _ in ranked_documents
            for sentence_index in document_to_sentences[document_index]
        ]
        ranked_sentences = sentence_bm25.rank(query, top_sentences, candidate_sentences)
        evidence = []
        for rank, (sentence_index, sentence_score) in enumerate(ranked_sentences, 1):
            sentence = sentences[sentence_index]
            evidence.append({
                "evidence_id": f"{sentence['doc_id']}:{sentence['sentence_id']}",
                "text": sentence["text"], "title": sentence["title"], "url": None,
                "source": "scifact_open_bm25",
                "metadata": {
                    "doc_id": sentence["doc_id"], "sentence_id": sentence["sentence_id"],
                    "sentence_rank": rank, "sentence_bm25": sentence_score,
                },
            })
        output.append({
            "example_id": f"scifact-{split}:{claim['id']}", "dataset": "SciFact",
            "split": split, "claim": query, "label": claim_label(claim),
            "evidence": evidence,
            "metadata": {
                "evidence_protocol": "open_corpus_bm25_top5_documents_top5_sentences",
                "retrieval_uses_cited_doc_ids": False,
                "retrieval_uses_gold_rationales": False,
                "retrieval_uses_gold_label": False,
                "retrieved_doc_ids": [documents[index]["doc_id"] for index, _ in ranked_documents],
                "retrieved_doc_scores": [score for _, score in ranked_documents],
            },
        })
        if progress_every and (claim_index % progress_every == 0 or claim_index == len(claim_rows)):
            elapsed = time.perf_counter() - started
            rate = claim_index / elapsed if elapsed else 0.0
            eta = (len(claim_rows) - claim_index) / rate if rate else 0.0
            print(
                f"progress={claim_index}/{len(claim_rows)} rate={rate:.2f}/s eta={eta:.1f}s",
                flush=True,
            )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--top-documents", type=int, default=5)
    parser.add_argument("--top-sentences", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    corpus, claims = list(read_jsonl(args.corpus)), list(read_jsonl(args.claims))
    rows = retrieve(
        corpus, claims, args.split, args.top_documents, args.top_sentences,
        args.progress_every,
    )
    manifest = {
        "dataset": "SciFact", "split": args.split,
        "evidence_protocol": "open_corpus_two_stage_bm25",
        "top_documents": args.top_documents, "top_sentences": args.top_sentences,
        "bm25": {"k1": 1.2, "b": 0.75},
        "rows": len(rows), "labels": dict(Counter(row["label"] for row in rows)),
        "evidence_count_distribution": dict(Counter(len(row["evidence"]) for row in rows)),
        "corpus_documents": len(corpus), "corpus_sha256": sha256_file(args.corpus),
        "claims_sha256": sha256_file(args.claims),
        "leakage_guards": {
            "uses_cited_doc_ids": False, "uses_gold_rationales": False,
            "uses_gold_labels_for_ranking": False,
        },
    }
    write_jsonl(args.output, rows)
    atomic_write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))
    print(f"SCIFACT_RETRIEVAL_READY rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
