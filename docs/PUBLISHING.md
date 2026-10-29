# Public Release Procedure

This checklist describes how to publish FHEViT as a standalone public GitHub
repository. The current working copy may live inside a larger local workspace; do
not publish the parent repository by accident.

## Pre-publish checks

From the FHEViT directory, confirm that the source-only checks pass:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --tb=short
.\.venv\Scripts\python.exe experiments\audit_workspace.py --check
.\.venv\Scripts\python.exe tools\public_release_check.py
```

Review the staged file list. It may contain source, tests, documentation, figures,
license files, and the small README indexes under `data/`, `models/`, `artifacts/`,
and `experiments/results/`. It must not contain data, images from a dataset,
patient-level metadata, model snapshots, checkpoints, feature caches, result JSON,
private-client state, virtual environments, build outputs, or absolute local paths.

## Reproducing the source release

The public clone reproduces installation, the package build, the test suite, code
map, and release audits without external data. Aggregate scientific results require
lawful access to the exact provider revisions, cohort manifests, model/checkpoint
revisions, and local feature caches documented in `data/README.md`,
`models/README.md`, `docs/ENVIRONMENT.md`, and `EVALUATION_PROTOCOL.md`.

## Citation and archival discovery

Use `CITATION.cff` when citing the software. Cite the original NIH ChestX-ray14,
OpenI, COVID dataset, and model papers separately as listed in `RELATED_WORK.md`.
For a citable archival release, enable GitHub's Zenodo integration after the public
repository and metadata have been reviewed; add the minted DOI to `CITATION.cff`
only after Zenodo issues it.
