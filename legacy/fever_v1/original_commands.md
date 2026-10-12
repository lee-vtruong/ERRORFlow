# Original FEVER command ledger

These are the canonical command shapes recovered from the experiment log.
Paths are server paths; large outputs are intentionally not in Git.

```bash
cd ~/whale/ERRORFlow
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src
```

## Retrieval and V0 baseline

```bash
python scripts/retrieve_fts.py \
  --db data/processed/fever/wiki_sentences.sqlite \
  --claims data/processed/fever/shared_task_dev_all_claims.jsonl \
  --output data/processed/fever/shared_task_dev_all_retrieved.jsonl \
  --k 5

python scripts/run_qwen3_baseline.py \
  --model /home/stackops/whale/cache/models/Qwen3-4B-Instruct-2507 \
  --input data/processed/fever/shared_task_dev_all_retrieved.jsonl \
  --output outputs/qwen3_baseline/shared_task_dev_all_confidence.jsonl \
  --split shared_task_dev_all

python scripts/evaluate_predictions.py \
  --gold data/processed/fever/shared_task_dev_all_retrieved.jsonl \
  --predictions outputs/qwen3_baseline/shared_task_dev_all_confidence.jsonl \
  --output outputs/qwen3_baseline/shared_task_dev_all_metrics.json
```

## V1 1K retrieval and reranking

```bash
python scripts/retrieve_fever_titles.py \
  --claims data/processed/fever/shared_task_dev_first1000_claims.jsonl \
  --wiki-dir data/raw/fever/wiki-pages \
  --output data/processed/fever/shared_task_dev_first1000_title_candidates50.jsonl \
  --k 50

python scripts/merge_retrieval_candidates.py \
  --lexical data/processed/fever/shared_task_dev_first1000_candidates50.jsonl \
  --title data/processed/fever/shared_task_dev_first1000_title_candidates50.jsonl \
  --output data/processed/fever/shared_task_dev_first1000_hybrid_candidates100.jsonl \
  --k 100 --title-first 50

python scripts/rerank_cross_encoder.py \
  --input data/processed/fever/shared_task_dev_first1000_hybrid_candidates100.jsonl \
  --output data/processed/fever/shared_task_dev_first1000_bge_top5.jsonl \
  --model /home/stackops/whale/cache/models/bge-reranker-v2-m3 \
  --top-k 5
```

## Evaluation and significance

```bash
python scripts/evaluate_predictions.py --gold GOLD.jsonl \
  --predictions PREDICTIONS.jsonl --output METRICS.json

python scripts/paired_bootstrap.py --gold GOLD.jsonl \
  --baseline BASELINE.jsonl --candidate CANDIDATE.jsonl \
  --output BOOTSTRAP.json --samples 5000 --seed 2026
```

Exact option sets evolved between commits. Before replaying, use the audited
commit and run each script with `--help`.
