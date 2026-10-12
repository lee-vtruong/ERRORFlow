# ERRORFlow

Training-free, error-memory-conditioned fact verification. ERRORFlow observes
baseline failures, diagnoses reusable error patterns, measures which corrective
actions helped, stores those outcomes, and conditionally reuses them under an
explicit effectiveness/cost policy.

## Project status

FEVER V0/V1 is preserved under `legacy/fever_v1/`. V2 now separates:

1. frozen baseline verification;
2. offline error diagnosis and structured memory construction;
3. intervention evaluation;
4. online adaptive routing;
5. leakage-safe evaluation and cost analysis;
6. canonical adapters for FEVER, AVeriTeC, SciFact, and Climate-FEVER;
7. config-hashed run manifests and resumable, atomic artifacts.

No V2 performance gain is claimed yet. New datasets and full model runs have
not been executed in this checkout.

See [`workrule.md`](workrule.md) for the local/server workflow and [`docs/WORK_LOG.md`](docs/WORK_LOG.md) for the required step-by-step record.

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

Normalize a dataset file without downloading or running a model:

```bash
PYTHONPATH=src python -m errorflow.cli normalize \
  --dataset scifact --split train \
  --input data/raw/scifact/claims_train.jsonl \
  --output data/processed/v2/scifact/train.jsonl
```

Build memory from paired outcomes:

```bash
PYTHONPATH=src python scripts/build_v2_error_memory.py \
  --input outputs/v2/paired_train_outcomes.jsonl \
  --output outputs/v2/memory/train.jsonl \
  --dataset SciFact --source-split train --action evidence_critic
```

The model, dataset, retrieval corpus and experiment outputs are intentionally not committed to Git.
