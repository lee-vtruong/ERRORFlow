# ERRORFlow V2 server handoff

## What to run immediately after pull

This stage is code verification only. It does not download datasets or launch a GPU job.

```bash
cd ~/whale/ERRORFlow
git pull --ff-only origin main
source ~/whale/GraphCURE/.venv/bin/activate
export PYTHONPATH=src
python -m pytest -q
python -m errorflow.cli --help
df -h /
```

Expected test count after Step 45: **66 passed**. A higher count is acceptable; a failure is not.

## Before the first V2 pilot

1. Keep at least 8–10 GB disk headroom; do not duplicate FEVER/model caches.
2. Locate or provide `Text_Only_Fact_Checking_Training_Free_Dataset_Survey.xlsx`.
3. Download only one primary dataset first (AVeriTeC is the default config), and record its official source, revision, split names, row counts, and checksum.
4. Normalize at most 25 examples with `configs/v2/averitec_smoke.yaml` semantics.
5. Build memory from **train only**. Do not pass `--permit-evaluation-routing` in a reported experiment.
6. Run a CPU/schema dry run before loading Qwen or an embedding model.

## Stop conditions

Stop and report rather than improvise if disk free space drops below 5 GB, the dataset license/source is ambiguous, official splits cannot be identified, a model revision differs from the manifest, or resume output contains duplicate example IDs.
