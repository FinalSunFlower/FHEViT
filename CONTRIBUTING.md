# Contributing to FHEViT

FHEViT is research software. Contributions should preserve the distinction between
the current plaintext InternVL2-2B evaluation line, retained historical baselines,
and the reduced CKKS prototype.

## Before opening a change

1. Read `README.md`, `docs/RESEARCH_NOTES.zh-CN.md`, and
   `docs/handoff/STATUS.md`.
2. Do not add datasets, patient identifiers, cohort manifests, model snapshots,
   checkpoints, feature caches, private ledgers, or generated result files.
3. Run the source-only checks from `docs/REPRODUCTION.md`.
4. For a scientific change, record the protocol, data revision, model revision,
   seed, and limitations in the relevant research note before interpreting a result.

## Required checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q --tb=short
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe experiments\audit_workspace.py --check
.\.venv\Scripts\python.exe tools\public_release_check.py
```

Keep pull requests focused. Explain whether a change affects a public API, a frozen
evaluation rule, a historical baseline, or only maintenance tooling. Never claim
clinical readiness, privacy, non-inferiority, or novelty beyond the evidence in
`docs/handoff/STATUS.md` and `docs/handoff/CLAIMS_AUDIT.md`.
