# Environment and Reproduction

This document describes the tested local baseline. It is deliberately separate
from the scientific protocol: changing a package version does not change a
locked result until the experiment is rerun and recorded as a new result.

## Tested baseline

| Component | Version |
|---|---|
| Python | 3.9.21 |
| PyTorch | 2.8.0+cu126 |
| torchvision | 0.23.0+cu126 |
| CUDA runtime reported by PyTorch | 12.6 |
| NumPy | 2.0.2 |
| SciPy | 1.13.1 |
| scikit-learn | 1.6.1 |
| TenSEAL | 0.3.16 |
| Pillow | 11.3.0 |

`torch.cuda.is_available()` was true in the recorded workstation. CPU-only
execution is sufficient for unit tests but is not expected to reproduce the
large-model or CKKS timings.

## Fresh setup (PowerShell)

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# Reproduce the recorded CUDA environment (omit for CPU-only tests):
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
# Install test and experiment dependencies:
.\.venv\Scripts\python.exe -m pip install -e ".[dev,experiments]"
# Optional LLM adapters (check model and bitsandbytes platform requirements first):
.\.venv\Scripts\python.exe -m pip install -e ".[dev,experiments,llm]"
```

The CUDA command matches the recorded PyTorch build; CUDA driver and GPU hardware
must also be compatible. CPU-only environments can run the tests but will not
reproduce accelerator timings. The repository uses dependency ranges, not a fully
resolved lockfile, so exact performance replication also requires recording the
resolved package versions, OS, GPU/driver, model revisions, and data hashes.

The package never downloads model snapshots, datasets, or private ledgers implicitly.
Find provider links, expected local paths, and preparation scripts in
[`../data/README.md`](../data/README.md) and [`../models/README.md`](../models/README.md).
The root [`../artifacts.manifest.json`](../artifacts.manifest.json) contains hashes,
not download links. Keep all assets in the ignored local directories below.

## Verification commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q --tb=short
.\.venv\Scripts\python.exe experiments\audit_workspace.py --check
.\.venv\Scripts\python.exe tools\public_release_check.py
```

`experiments/results/` contains local run outputs and is not a clean-room
reproduction input. The authoritative scientific values are the JSON fields
referenced by `docs/handoff/STATUS.md`; do not silently recompute a different
split or open the reserved confirmation cohort.

## What a Clone Can Reproduce

A source-only clone can install the package, build a wheel, run the 139 unit tests,
and inspect every maintained experiment script. It cannot reproduce the published
aggregate study results from source alone: authorized NIH data, local cohort
manifests, exact model/checkpoint revisions, and cached features are intentionally
not redistributed. Some scripts also require earlier local result manifests as
protocol inputs. Follow the current status page and frozen protocol in order; do not
run sealed confirmation or holdout scripts to make a clone appear complete. This is
a deliberate data-governance boundary, not a missing Git submodule.

## Local-only assets

The following are intentionally not part of a public source release:

- `data/`: downloaded radiographs, cohort manifests, caches, and patient-level
  identifiers;
- `models/`: external model snapshots;
- `artifacts/`: trained weights and derived checkpoints;
- `experiments/results/`: some outputs contain patient IDs, paths, or raw text;
- `private-client/`: persistent private release state;
- `.venv/`, `build/`, `dist/`, `*.egg-info/`, and audit work snapshots.

The root `artifacts.manifest.json` records hashes for locally available weights
without redistributing those weights. Do not commit a local absolute path or a
dataset archive to make a reproduction appear self-contained.

The corresponding source locations and provider links are indexed in
[`../data/README.md`](../data/README.md), [`../models/README.md`](../models/README.md),
and [`../artifacts/README.md`](../artifacts/README.md). Those Markdown indexes are
the only files intentionally retained in the otherwise local-only asset folders.
