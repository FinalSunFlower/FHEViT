# Reproduction Guide

This guide separates what a clean source clone can reproduce from results that require
restricted external inputs. It is intentionally explicit: a missing local dataset or
checkpoint is not filled with a substitute, and a public release never contains
patient-level data.

## 1. Clean source validation

From a fresh clone of this project:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,experiments]"
.\.venv\Scripts\python.exe -m pytest -q --tb=short
.\.venv\Scripts\python.exe experiments\audit_workspace.py --check
.\.venv\Scripts\python.exe tools\public_release_check.py
```

This validates the package, synthetic/unit tests, code map, local links, artifact
manifest format, and source-only release boundary. It does not download data or
model weights. Use `docs/ENVIRONMENT.md` for the recorded CUDA variant and exact
environment notes.

## 2. Provision inputs locally

Read [`../data/README.md`](../data/README.md) before downloading anything. It maps
each data product to its provider, local directory, and preparation script. Read
[`../models/README.md`](../models/README.md) for model-card URLs and local snapshot
names. Obtain all data and models under their own terms; do not commit them.

The minimal OpenI quickstart layout is:

```text
data/openi_subset/metadata.json
data/openi_subset/images/<image files>
```

The NIH preparation starts from the label table and writes selected image bytes and
manifests beneath the ignored `data/` directory:

```powershell
.\.venv\Scripts\python.exe experiments\fetch_nih_probe.py
```

The remaining NIH cohort scripts are protocol-gated. Run them only in the order and
with the exact exclusions specified by `EVALUATION_PROTOCOL.md`. Never open the
reserved confirmation or final holdout merely to satisfy a missing-file check.

## 3. Provision models and artifacts locally

The LLM adapters load from local paths such as `models/internvl2-2b/` and
`models/qwen2vl-2b/`; they do not download at runtime. The trained adapter and encoder
files belong under `artifacts/` and are checked against the local
`artifacts.manifest.json`. That manifest records SHA-256 values only. A matching hash
does not grant redistribution rights and is not a download source.

The historical aggregate experiments also expect local cohort manifests, native-model
outputs, and feature caches under `data/` and `experiments/results/`. Those files are
not clean-room inputs and may contain patient IDs, paths, report text, or medical
representations. They are intentionally not reconstructed from public summaries.

## 4. Reproduce a reported result

Before running a scientific script, verify all of the following locally:

1. The dataset provider, revision/archive hash, and local manifest match
   `data/README.md` and the relevant protocol section.
2. The model-card revision, adapter checkpoint hash, and preprocessing contract match
   the run metadata.
3. The patient split, seed, endpoint, bootstrap rule, and exclusion list match
   `EVALUATION_PROTOCOL.md`.
4. The confirmation and final holdout flags remain closed unless the current status
   page explicitly authorizes a new protocol.

The current scientific line is the plaintext InternVL2-2B prefix study. The CKKS
measurement is a reduced block, not an encrypted full InternViT tower. Use
`docs/handoff/STATUS.md` and `docs/handoff/CLAIMS_AUDIT.md` to interpret any aggregate
result; do not turn a local result JSON into a public patient-level artifact.

## 5. Public citation

Cite this software using [`../CITATION.cff`](../CITATION.cff), and cite the original
dataset and model papers listed in [`../RELATED_WORK.md`](../RELATED_WORK.md). The
public release already records the software author, version, release date, license,
and repository URL in `CITATION.cff`; update those fields only when preparing a new
version or adding a verified archival DOI.
