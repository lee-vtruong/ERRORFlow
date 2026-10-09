"""Retrieve top-k passages from the independent FEVER FTS index."""
import argparse, json, re, sqlite3
from pathlib import Path

def fts_query(text):
    tokens = re.findall(r"[\w]+", text, flags=re.UNICODE)
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens) or '""'

def retrieve(con, claim, k):
    rows = con.execute("SELECT text FROM sentences_fts WHERE sentences_fts MATCH ? ORDER BY bm25(sentences_fts) LIMIT ?", (fts_query(claim), k)).fetchall()
    return [row[0] for row in rows]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--db", type=Path, required=True); p.add_argument("--claims", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--k", type=int, default=5); a = p.parse_args()
    con = sqlite3.connect(a.db); a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.claims.open(encoding="utf-8") as source, a.output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip(): continue
            row = json.loads(line); evidence = retrieve(con, row["claim"], a.k)
            target.write(json.dumps({"claim_id": row["claim_id"], "claim": row["claim"], "evidence": evidence, "gold_label": row.get("gold_label"), "dataset": row.get("dataset", "FEVER"), "dataset_version": row.get("dataset_version", "1.0"), "source_split": row.get("source_split", "unknown"), "retrieval": {"method": "sqlite_fts5_bm25", "k": a.k}}, ensure_ascii=False) + "\n")
    con.close()

if __name__ == "__main__": main()
