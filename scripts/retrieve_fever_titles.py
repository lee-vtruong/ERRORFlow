"""Retrieve FEVER sentences from claim-matched Wikipedia page titles."""
import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were",
    "with",
}


def words(text):
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", str(text))


def normalized_key(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def page_title_keys(page_id):
    title = str(page_id).replace("_", " ")
    title = title.replace("-LRB-", "(").replace("-RRB-", ")")
    keys = {normalized_key(title)}
    base = re.sub(r"\s*\([^)]*\)\s*$", "", title)
    if base:
        keys.add(normalized_key(base))
    return {key for key in keys if key}


def claim_title_keys(claim, max_ngram=6):
    tokens = words(claim)
    keys = set()
    for size in range(2, min(max_ngram, len(tokens)) + 1):
        for start in range(len(tokens) - size + 1):
            span = tokens[start:start + size]
            if any(token.lower() not in STOPWORDS for token in span):
                keys.add(normalized_key(" ".join(span)))
    for token in tokens:
        if len(token) > 2 and token[0].isupper() and token.lower() not in STOPWORDS:
            keys.add(normalized_key(token))
    return keys


def parse_page_sentences(page):
    records = []
    for line in str(page.get("lines", "")).splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0].isdigit() and parts[1].strip():
            records.append((int(parts[0]), parts[1]))
    return records


def sentence_score(claim, sentence, sentence_id):
    claim_terms = {token.lower() for token in words(claim) if token.lower() not in STOPWORDS}
    sentence_terms = {token.lower() for token in words(sentence) if token.lower() not in STOPWORDS}
    overlap = len(claim_terms & sentence_terms)
    cosine_like = overlap / math.sqrt(max(1, len(claim_terms) * len(sentence_terms)))
    return (cosine_like, overlap, -sentence_id)


def retrieve_title_candidates(claim_rows, wiki_dir, k=50, max_ngram=6, progress_every=10):
    claims = {str(row["claim_id"]): row for row in claim_rows}
    key_to_claims = defaultdict(set)
    for claim_id, row in claims.items():
        for key in claim_title_keys(row["claim"], max_ngram):
            key_to_claims[key].add(claim_id)

    candidates = defaultdict(dict)
    shards = sorted(wiki_dir.glob("*.jsonl"))
    started = time.perf_counter()
    matched_pages = 0
    for shard_index, shard in enumerate(shards, start=1):
        with shard.open(encoding="utf-8") as source:
            for raw in source:
                page = json.loads(raw)
                matching_claims = set()
                for key in page_title_keys(page.get("id", "")):
                    matching_claims.update(key_to_claims.get(key, ()))
                if not matching_claims:
                    continue
                matched_pages += 1
                page_id = str(page.get("id", ""))
                for sentence_id, text in parse_page_sentences(page):
                    for claim_id in matching_claims:
                        key = (page_id, sentence_id)
                        score = sentence_score(claims[claim_id]["claim"], text, sentence_id)
                        candidates[claim_id][key] = (score, text)
        if shard_index == 1 or shard_index % progress_every == 0 or shard_index == len(shards):
            elapsed = time.perf_counter() - started
            print(
                f"progress_shards={shard_index}/{len(shards)} matched_pages={matched_pages} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    results = {}
    for claim_id in claims:
        ranked = sorted(
            candidates[claim_id].items(),
            key=lambda item: (item[1][0], item[0][0]),
            reverse=True,
        )[:k]
        results[claim_id] = [
            {
                "page_id": page_id,
                "sentence_id": sentence_id,
                "text": value[1],
                "title_score": value[0][0],
                "token_overlap": value[0][1],
            }
            for (page_id, sentence_id), value in ranked
        ]
    return results, matched_pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-dir", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--max-ngram", type=int, default=6)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    with args.claims.open(encoding="utf-8") as source:
        claim_rows = [json.loads(line) for line in source if line.strip()]
    results, matched_pages = retrieve_title_candidates(
        claim_rows, args.wiki_dir, args.k, args.max_ngram, args.progress_every
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    found = 0
    with args.output.open("w", encoding="utf-8") as target:
        for row in claim_rows:
            records = results[str(row["claim_id"])]
            found += bool(records)
            metadata = [
                {
                    "page_id": record["page_id"],
                    "sentence_id": record["sentence_id"],
                    "rank": rank,
                    "title_score": record["title_score"],
                    "token_overlap": record["token_overlap"],
                    "source": "exact_claim_title",
                }
                for rank, record in enumerate(records, start=1)
            ]
            output = dict(row)
            output["evidence"] = [record["text"] for record in records]
            output["evidence_metadata"] = metadata
            output["retrieval"] = {
                "method": "exact_claim_ngram_page_title",
                "max_ngram": args.max_ngram,
                "k": args.k,
                "provenance": True,
            }
            target.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(
        f"TITLE_RETRIEVAL_DONE rows={len(claim_rows)} evidence_found={found} "
        f"matched_pages={matched_pages} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
