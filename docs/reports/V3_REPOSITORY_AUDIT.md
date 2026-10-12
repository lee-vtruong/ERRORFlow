# ERRORFlow V3 repository audit

Status: **COMPLETED** on 2026-10-12.

## Preserved boundary

- V2 HEAD: `6e20716bba77e2217f8532e717857318a017395c`.
- Annotated and pushed tag: `archive/errorflow-v2-conflict-audit`.
- Historical B1 profile: `conflict_audit_v1`.
- Exact B1 UTF-8 length: 372 bytes.
- B1 SHA-256: `700ebdc1b55cb9f04f7524f31b7a5d3bd41813f198e5629fe75ca551333decf7`.
- Historical reports remain under `docs/STEP_48_V2_HOLDOUT_ONE_SHOT.md` and
  `docs/reports/AVERITEC_V2_HOLDOUT_REPORT.md`.
- FEVER archives and all V1/V2 scripts remain in place.

The existing dirty working-tree files predated V3 and were not staged or
modified by the archive operation.

## Contamination boundary

AVeriTeC IDs `dev:25` through `dev:499` are opened historical evaluation data.
V3 development code rejects them. Development is restricted to official train
data, grouped folds, and fold-excluded memory retrieval. The opened holdout may
be cited historically but cannot select prompts, thresholds, memory types, or
router parameters.

## Training-free terminology

V3 freezes Qwen3-4B-Instruct-2507 and performs no gradient updates. It is
parameter-training-free, but prompts are train-selected and memories use labeled
train outcomes. Any fitted or selected routing policy must be described
separately rather than as fully zero-shot.

## Reproduction checks

```bash
python -m errorflow.cli doctor
python -m errorflow.cli v3 audit --output outputs/v3/repository_audit.json
git rev-list -n 1 archive/errorflow-v2-conflict-audit
```
