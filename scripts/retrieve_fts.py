"""Retrieve top-k passages from the independent FEVER FTS index."""
import argparse, json, re, sqlite3, time
from pathlib import Path

def fts_query(text, max_tokens=8, operator=" OR "):
    tokens = re.findall(r"[\w]+", text, flags=re.UNICODE)[:max_tokens]
    return operator.join('"' + token.replace('"', '""') + '"' for token in tokens) or '""'

def retrieve(con, claim, k, mode="and_or", max_tokens=8):
    tokens = re.findall(r"[\w]+", claim, flags=re.UNICODE)[:max_tokens]
    quoted = ['"' + token.replace('"', '""') + '"' for token in tokens]
    queries = [" AND ".join(quoted)] if mode == "and_or" else []
    queries.append(" OR ".join(quoted))
    for query in queries:
        if not query: continue
        rows = con.execute("SELECT text FROM sentences_fts WHERE sentences_fts MATCH ? ORDER BY bm25(sentences_fts) LIMIT ?", (query, k)).fetchall()
        if len(rows) >= k or query == queries[-1]: return [row[0] for row in rows]
    return []

def main():
    p=argparse.ArgumentParser(); p.add_argument("--db",type=Path,required=True); p.add_argument("--claims",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--k",type=int,default=5); p.add_argument("--progress-every",type=int,default=25); p.add_argument("--mode",choices=["and_or","or"],default="and_or"); p.add_argument("--max-query-tokens",type=int,default=8); a=p.parse_args()
    con=sqlite3.connect(a.db); a.output.parent.mkdir(parents=True,exist_ok=True); total=sum(1 for line in a.claims.open(encoding="utf-8") if line.strip()); done=found=0; started=time.perf_counter()
    print(f"RETRIEVAL_START total={total} k={a.k} mode={a.mode} db={a.db}",flush=True)
    with a.claims.open(encoding="utf-8") as source,a.output.open("w",encoding="utf-8") as target:
        for line in source:
            if not line.strip(): continue
            row=json.loads(line); evidence=retrieve(con,row["claim"],a.k,a.mode,a.max_query_tokens); done+=1; found+=bool(evidence)
            target.write(json.dumps({"claim_id":row["claim_id"],"claim":row["claim"],"evidence":evidence,"gold_label":row.get("gold_label"),"dataset":row.get("dataset","FEVER"),"dataset_version":row.get("dataset_version","1.0"),"source_split":row.get("source_split","unknown"),"retrieval":{"method":"sqlite_fts5_bm25","mode":a.mode,"max_query_tokens":a.max_query_tokens,"k":a.k}},ensure_ascii=False)+"\n")
            if done==1 or done%a.progress_every==0 or done==total:
                elapsed=time.perf_counter()-started; rate=done/elapsed if elapsed else 0; eta=(total-done)/rate if rate else 0; print(f"progress={done}/{total} evidence_found={found} rate={rate:.2f}/s eta={eta:.1f}s",flush=True)
    con.close(); print(f"RETRIEVAL_DONE rows={done} evidence_found={found} elapsed_s={time.perf_counter()-started:.1f}",flush=True)

if __name__=="__main__": main()
