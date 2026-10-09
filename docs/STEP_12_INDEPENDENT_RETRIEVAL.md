# Step 12 — Independent retrieval baseline

Mục tiêu: query evidence bằng claim trên toàn bộ FEVER wiki-pages, không dùng gold evidence. Index dùng SQLite FTS5 và truy vấn BM25.

Push code rồi trên server chạy trong tmux:

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src
mkdir -p data/processed/fever
tmux new -s errorflow-index
python scripts/build_wiki_fts.py \
  --wiki-dir data/raw/fever/wiki-pages \
  --db data/processed/fever/wiki_sentences.sqlite
```

Detach bằng `Ctrl-b`, `d`. Chờ dòng `INDEX_READY`. Sau đó:

```bash
python scripts/retrieve_fts.py \
  --db data/processed/fever/wiki_sentences.sqlite \
  --claims data/processed/fever/train_first1000.jsonl \
  --output data/processed/fever/train_first1000_retrieved.jsonl \
  --k 5
```

Artifact index có thể rất lớn và không commit vào Git.

Script hiển thị `RETRIEVAL_START`, progress theo claim, tốc độ, ETA và `RETRIEVAL_DONE`. Có thể chỉnh tần suất bằng `--progress-every 10`.

Nếu SQLite báo `no such column` khi query claim có tên riêng hoặc dấu câu, hãy pull bản mới: query builder phải quote token trước khi gửi vào FTS5.
