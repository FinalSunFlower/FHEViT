# FHEViT

Research code for studying privacy and utility at the interface between chest
radiographs, visual-language-model representations, and reduced CKKS circuits.
The repository is an empirical research artifact, not a clinical product or a
claim that encrypted inference preserves image privacy.

## Start here

- [Chinese research notes](docs/RESEARCH_NOTES.zh-CN.md): implementation map,
  architecture, results, pitfalls, and publication checklist.
- [Environment and reproduction](docs/ENVIRONMENT.md): tested versions and
  commands for a clean local environment.
- [Reproduction guide](docs/REPRODUCTION.md): source-only validation, lawful local
  data/model provisioning, and the limits of aggregate-result reproduction.
- [Data sources and local layout](data/README.md): official provider links,
  expected local paths, preparation scripts, and the public-release boundary.
- [Model snapshots](models/README.md): provider model pages and exact local
  directories; model files are never bundled.
- [Current status](docs/handoff/STATUS.md): the only authoritative summary of
  the current scientific conclusions.
- [Code map](CODE_MAP.md): every maintained Python module under `fpavlm/`,
  `experiments/`, `tests/`, and `tools/`.
- [Evaluation protocol](EVALUATION_PROTOCOL.md): frozen cohort, split, scoring,
  bootstrap, and non-inferiority rules.
- [Related work](RELATED_WORK.md): selected primary literature and a bounded
  novelty statement.
- [Public release procedure](docs/PUBLISHING.md): standalone GitHub publication,
  staged-file review, and archival citation steps.

## What is implemented

The package contains deterministic preprocessing, representation contracts,
patient-level evaluation and privacy audits, affine and spatial release
prototypes, a reduced polynomial CKKS block implemented with TenSEAL, and
optional InternVL2/Qwen2-VL adapters. The experiment scripts implement the
NIH ChestX-ray14 cohort construction, PCA-prefix utility tests, reconstruction
attacks, pixel-student capacity tests, and CKKS depth/sanity measurements.

The current scored line is a plaintext InternVL2-2B prefix study. The measured
CKKS object is a reduced block, not a complete encrypted InternViT tower. The
older interactive refresh and adapter code remain as clearly labeled research
baselines because tests and historical measurements still depend on them.

## Current evidence

On the development analysis set, a near-full-rank, zero-coordinate-noise PCA
release passes the pre-registered PA-only utility rule. Coordinate noise at the
tested levels fails. The reconstruction audit recovers chest-radiograph
structure from rank-1024 and rank-1999 zero-noise releases; no pixel student at
depth at most five passes; and the locked polynomial tower becomes non-finite
in block two. The reserved confirmation cohort remains closed.

These results do not establish privacy, clinical equivalence, losslessness,
state-of-the-art performance, or firstness. See the status and claims audit
before quoting a number.

## Verification

From the project root on Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --tb=short
.\.venv\Scripts\python.exe experiments\audit_workspace.py --check
.\.venv\Scripts\python.exe tools\public_release_check.py
```

The current local baseline is 139 passing tests. A source clone reproduces the
package and tests without data, but not the reported aggregate study results:
those require lawful access to the exact dataset, cohort manifests, model and
checkpoint revisions, and local feature caches. Large datasets, model snapshots,
patient-level outputs, private ledgers, caches, and trained weights are intentionally
excluded from a source release; see `.gitignore`, [data/README.md](data/README.md),
and [ENVIRONMENT.md](docs/ENVIRONMENT.md).

## Citation and license

Use [CITATION.cff](CITATION.cff) for software citation. The project is released
under the Apache License 2.0; dataset and pretrained-model terms remain those
of their original providers.
