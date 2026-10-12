"""Retrieve top-k passages from the independent FEVER FTS index."""
import argparse, json, re, sqlite3, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def fts_query(text, max_tokens=8, operator=" OR "):
    tokens = re.findall(r"[\w]+", text, flags=re.UNICODE)[:max_tokens]
    return operator.join('"' + token.replace('"', '""') + '"' for token in tokens) or '""'

def _queries(claim, mode="and_or", max_tokens=8):
    tokens = re.findall(r"[\w]+", claim, flags=re.UNICODE)[:max_tokens]
    quoted = ['"' + token.replace('"', '""') + '"' for token in tokens]
    queries = [" AND ".join(quoted)] if mode == "and_or" else []
    queries.append(" OR ".join(quoted))
    return [query for query in queries if query]

def retrieve(con, claim, k, mode="and_or", max_tokens=8):
    queries = _queries(claim, mode, max_tokens)
    for query in queries:
        rows = con.execute("SELECT text FROM sentences_fts WHERE sentences_fts MATCH ? ORDER BY bm25(sentences_fts), sentences_fts.rowid LIMIT ?", (query, k)).fetchall()
        if len(rows) >= k or query == queries[-1]: return [row[0] for row in rows]
    return []

def retrieve_records(con, claim, k, mode="and_or", max_tokens=8):
    """Return ranked evidence with FEVER page/sentence provenance."""
    queries = _queries(claim, mode, max_tokens)
    sql = """
        SELECT sentences.page_id, sentences.sentence_id, sentences.text,
               bm25(sentences_fts) AS bm25_score
        FROM sentences_fts
        JOIN sentences ON sentences.rowid = sentences_fts.rowid
        WHERE sentences_fts MATCH ?
        ORDER BY bm25_score, sentences_fts.rowid
        LIMIT ?
    """
    for query in queries:
        rows = con.execute(sql, (query, k)).fetchall()
        if len(rows) >= k or query == queries[-1]:
            return [
                {
                    "page_id": row[0],
                    "sentence_id": int(row[1]),
                    "text": row[2],
                    "rank": rank,
                    "bm25_score": float(row[3]),
                }
                for rank, row in enumerate(rows, start=1)
            ]
    return []

def main():
    p=argparse.ArgumentParser(); p.add_argument("--db",type=Path,required=True); p.add_argument("--claims",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--k",type=int,default=5); p.add_argument("--progress-every",type=int,default=25); p.add_argument("--mode",choices=["and_or","or"],default="and_or"); p.add_argument("--max-query-tokens",type=int,default=8); p.add_argument("--workers",type=int,default=1); p.add_argument("--resume",action="store_true"); a=p.parse_args()
    all_rows=[json.loads(line) for line in a.claims.open(encoding="utf-8") if line.strip()]; a.output.parent.mkdir(parents=True,exist_ok=True); completed=set()
    if a.resume and a.output.exists(): completed={str(json.loads(line)["claim_id"]) for line in a.output.open(encoding="utf-8") if line.strip()}
    rows=[row for row in all_rows if str(row["claim_id"]) not in completed]; started=time.perf_counter(); found=0
    print(f"RETRIEVAL_START total={len(all_rows)} completed={len(completed)} remaining={len(rows)} k={a.k} mode={a.mode} workers={a.workers} db={a.db}",flush=True)
    def work(row):
        con=sqlite3.connect(f"file:{a.db}?mode=ro", uri=True); records=retrieve_records(con,row["claim"],a.k,a.mode,a.max_query_tokens); con.close(); return row,records
    output_mode="a" if a.resume and completed else "w"
    with ThreadPoolExecutor(max_workers=a.workers) as pool, a.output.open(output_mode,encoding="utf-8") as target:
        for done,(row,records) in enumerate(pool.map(work,rows),start=1):
            evidence=[record["text"] for record in records]
            metadata=[{key:record[key] for key in ("page_id","sentence_id","rank","bm25_score")} for record in records]
            found+=bool(evidence); target.write(json.dumps({"claim_id":row["claim_id"],"claim":row["claim"],"evidence":evidence,"evidence_metadata":metadata,"gold_label":row.get("gold_label"),"dataset":row.get("dataset","FEVER"),"dataset_version":row.get("dataset_version","1.0"),"source_split":row.get("source_split","unknown"),"retrieval":{"method":"sqlite_fts5_bm25","mode":a.mode,"max_query_tokens":a.max_query_tokens,"k":a.k,"workers":a.workers,"provenance":True}},ensure_ascii=False)+"\n")
            target.flush()
            if done==1 or done%a.progress_every==0 or done==len(rows):
                elapsed=time.perf_counter()-started; rate=done/elapsed if elapsed else 0; eta=(len(rows)-done)/rate if rate else 0; print(f"progress={len(completed)+done}/{len(all_rows)} session_done={done}/{len(rows)} evidence_found={found} rate={rate:.2f}/s eta={eta:.1f}s",flush=True)
    print(f"RETRIEVAL_DONE total_rows={len(all_rows)} resumed_from={len(completed)} session_rows={len(rows)} evidence_found={found} elapsed_s={time.perf_counter()-started:.1f}",flush=True)

if __name__=="__main__": main()
