# Step 08 — FEVER evidence resolution

## Đã làm

- `src/errorflow/fever_evidence.py` trích xuất `(wiki_page_id, sentence_id)` từ raw FEVER evidence groups.
- Có thể resolve sentence text từ FEVER `wiki-pages` JSONL records.
- Chưa tải wiki corpus tự động vì archive lớn: nguồn chính thức mô tả `wiki-pages.zip` khoảng 1.7 GB compressed và khoảng 7.25 GB extracted. [FEVER dataset](https://fever.ai/dataset/fever.html)

## Kiểm tra corpus trên server

```bash
find ~/whale/GraphCURE ~/whale/cache ~/whale/ERRORFlow/data \
  -type f \( -name 'wiki-pages*.jsonl' -o -name 'wiki-pages.zip' \) \
  2>/dev/null | head -50
```

Nếu chưa có corpus, tải vào vùng data chưa track Git:

```bash
mkdir -p ~/whale/ERRORFlow/data/raw/fever
wget -O ~/whale/ERRORFlow/data/raw/fever/wiki-pages.zip \
  https://fever.ai/download/fever/wiki-pages.zip
unzip -q ~/whale/ERRORFlow/data/raw/fever/wiki-pages.zip \
  -d ~/whale/ERRORFlow/data/raw/fever/
```

Không commit archive hoặc extracted Wikipedia corpus vào Git.
