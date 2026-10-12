import sqlite3
from importlib.machinery import SourceFileLoader

def test_retrieval_returns_match():
    module = SourceFileLoader("retrieve_fts", "scripts/retrieve_fts.py").load_module()
    con = sqlite3.connect(":memory:"); con.execute("CREATE VIRTUAL TABLE sentences_fts USING fts5(text)"); con.execute("INSERT INTO sentences_fts(text) VALUES ('Paris is the capital of France')")
    assert module.retrieve(con, "capital France", 1)


def test_retrieval_records_include_provenance_and_score():
    module = SourceFileLoader("retrieve_fts_records", "scripts/retrieve_fts.py").load_module()
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE sentences (rowid INTEGER PRIMARY KEY, page_id TEXT, sentence_id INTEGER, text TEXT)")
    con.execute("CREATE VIRTUAL TABLE sentences_fts USING fts5(text, content='sentences', content_rowid='rowid')")
    con.execute("INSERT INTO sentences(page_id,sentence_id,text) VALUES ('Paris',0,'Paris is the capital of France')")
    con.execute("INSERT INTO sentences_fts(sentences_fts) VALUES ('rebuild')")
    records = module.retrieve_records(con, "capital France", 1, mode="or")
    assert records[0]["page_id"] == "Paris"
    assert records[0]["sentence_id"] == 0
    assert records[0]["rank"] == 1
    assert isinstance(records[0]["bm25_score"], float)


def test_retrieval_quotes_fts_tokens():
    module = SourceFileLoader("retrieve_fts_safe", "scripts/retrieve_fts.py").load_module()
    assert '"Waldau"' in module.fts_query("Waldau (film) — 2005")
