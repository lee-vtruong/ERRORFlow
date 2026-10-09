import sqlite3
from importlib.machinery import SourceFileLoader

def test_retrieval_returns_match():
    module = SourceFileLoader("retrieve_fts", "scripts/retrieve_fts.py").load_module()
    con = sqlite3.connect(":memory:"); con.execute("CREATE VIRTUAL TABLE sentences_fts USING fts5(text)"); con.execute("INSERT INTO sentences_fts(text) VALUES ('Paris is the capital of France')")
    assert module.retrieve(con, "capital France", 1)
