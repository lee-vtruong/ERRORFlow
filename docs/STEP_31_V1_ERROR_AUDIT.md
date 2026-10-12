# Step 31 — Audit B1 errors before rebuilding ERRORFlow

## B1 result

On the same 1.000 balanced shared-task dev claims:

| Metric | B0 lexical | B1 stronger retrieval | Delta |
|---|---:|---:|---:|
| Accuracy | 55.60% | 58.30% | +2.70 |
| Macro-F1 | 54.07% | 53.36% | -0.71 |
| SUPPORTED F1 | 64.32% | 69.49% | +5.17 |
| REFUTED F1 | 60.59% | 64.08% | +3.49 |
| NEI F1 | 37.29% | 26.51% | -10.78 |

The net `+27` correct predictions comprise `+57` SUPPORTED correct, `+15` REFUTED correct, and `-45` NEI correct. B1 retrieves much better evidence but causes a relevance-as-entailment bias: NEI→SUPPORTED grows from 107 to 144 and NEI→REFUTED grows from 121 to 129.

## Build auditable B1 errors

```bash
python scripts/build_error_list.py \
  --gold data/processed/fever/shared_task_dev_first1000_bge_top5.jsonl \
  --predictions outputs/qwen3_baseline/shared_task_dev_first1000_v1_bge_top5.jsonl \
  --output outputs/qwen3_baseline/shared_task_dev_first1000_v1_errors.jsonl

python scripts/diagnose_errors.py \
  --input outputs/qwen3_baseline/shared_task_dev_first1000_v1_errors.jsonl \
  --output outputs/qwen3_baseline/shared_task_dev_first1000_v1_diagnosed.jsonl

python scripts/summarize_error_list.py \
  --input outputs/qwen3_baseline/shared_task_dev_first1000_v1_errors.jsonl \
  --output outputs/qwen3_baseline/shared_task_dev_first1000_v1_error_summary.json
```

Expected error count is `417`. The summary records label-pair counts, mean baseline confidence, confidence bins and whether title or lexical retrieval supplied the top-ranked evidence.

## Decision after audit

Do not copy V0's `REFUTED and confidence <= 0.90` router. B1's dominant failure is expected to be NEI overcommitment across both SUPPORTED and REFUTED predictions. Intervention design and routing must be selected from B1 recovery and regression measurements.
