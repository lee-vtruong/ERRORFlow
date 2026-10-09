# ERRORFlow

Error-Conditioned Memory and Adaptive Workflow Optimization for Training-Free Fact Verification.

## Project status

The repository is at the skeleton stage. The initial implementation will separate:

1. frozen baseline verification;
2. offline error diagnosis and structured memory construction;
3. intervention evaluation;
4. online adaptive routing;
5. leakage-safe evaluation and cost analysis.

See [`workrule.md`](workrule.md) for the local/server workflow and [`docs/WORK_LOG.md`](docs/WORK_LOG.md) for the required step-by-step record.

## Development

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

The model, dataset, retrieval corpus and experiment outputs are intentionally not committed to Git.
