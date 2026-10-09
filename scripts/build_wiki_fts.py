"""Build a SQLite FTS5 sentence index from FEVER wiki-page shards."""
import argparse, json, sqlite3
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wiki-dir", type=Path, required=True)
    p.add_argument("--db", type=Path, required=True)
    a = p.parse_args(); a.db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(a.db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("CREATE TABLE IF NOT EXISTS sentences (rowid INTEGER PRIMARY KEY, page_id TEXT, sentence_id INTEGER, text TEXT)")
    con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS sentences_fts USING fts5(text, content='sentences', content_rowid='rowid')")
    count = 0
    for shard in sorted(a.wiki_dir.glob("*.jsonl")):
        batch = []
        with shard.open(encoding="utf-8") as f:
            for raw in f:
                page = json.loads(raw)
                for line in str(page.get("lines", "")).splitlines():
                    parts = line.split("\t", 1)
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].strip():
                        batch.append((page.get("id", ""), int(parts[0]), parts[1]))
                    if len(batch) >= 5000:
                        con.executemany("INSERT INTO sentences(page_id,sentence_id,text) VALUES (?,?,?)", batch); count += len(batch); batch.clear()
        if batch:
            con.executemany("INSERT INTO sentences(page_id,sentence_id,text) VALUES (?,?,?)", batch); count += len(batch)
        con.commit(); print(f"indexed={count} shard={shard.name}", flush=True)
    con.execute("INSERT INTO sentences_fts(sentences_fts) VALUES ('rebuild')"); con.commit(); con.close()
    print(f"INDEX_READY rows={count} db={a.db}")

if __name__ == "__main__": main()
