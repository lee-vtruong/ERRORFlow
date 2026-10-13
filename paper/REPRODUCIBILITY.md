# Reproducing the ERRORFlow V3 paper

## What a clean clone can reproduce

```bash
python paper/scripts/build_figures.py
python paper/scripts/check_citations.py
pytest -q
```

Compile with the user-supplied Springer LNCS class available at `Template-Paper/`:

```bash
cd paper
TEXINPUTS=../Template-Paper: latexmk -pdf -interaction=nonstopmode -halt-on-error errorflow_v3.tex
```

On Windows PowerShell, set `$env:TEXINPUTS = "..\Template-Paper;"` before `latexmk`.

## Environment used for reported runs

- Python 3.11.15
- PyTorch 2.13.0+cu130
- Transformers 5.15.0
- PEFT 0.20.0
- NVIDIA GeForce RTX 5090, CUDA 13.0
- model identifier: `Qwen/Qwen3-4B-Instruct-2507`
- seed: 2026

The run manifest hashes the model identifier, not model weight bytes. A release archive must additionally pin the Hugging Face revision and checksum weight/tokenizer files.

## Core experiment commands

Prepare normalized datasets using the adapters in `src/errorflow/adapters/`. Score B1 and A3 with:

```bash
python scripts/run_v3_qwen_action.py --help
```

Materialize outcome memory and R6 policy with:

```bash
python scripts/build_v3_residual_memory.py --help
python scripts/materialize_v3_utility_policy.py --help
```

Run grouped source and matched-budget evaluations with:

```bash
python scripts/evaluate_v3_grouping_sensitivity.py --help
python scripts/evaluate_v3_matched_budget.py --help
```

Prepare/evaluate open-corpus SciFact evidence with:

```bash
python scripts/prepare_scifact_retrieved.py --help
python scripts/evaluate_scifact_retrieval.py --help
python scripts/evaluate_v3_external_transfer.py --help
```

Exact server commands depend on non-redistributed dataset/model paths. The claim ledger maps every reported value to the expected output.

## Protocol invariants

- Mean candidate-token log probability; one forward pass per label.
- Label order resolves exact ties.
- Five memory neighbors, minimum support two, cost weight 0.1, route iff utility is strictly positive.
- AVeriTeC source evaluation excludes the full 0.90-Jaccard claim group/fold.
- External routing uses only frozen AVeriTeC train memory and never target gold labels.
- SciFact `CONFLICTING -> NOT ENOUGH INFO` projection is post-routing and is reported alongside raw sensitivity.
