# Data Sources and Local Layout

中文索引见 [README.zh-CN.md](README.zh-CN.md)。

No dataset, patient list, cohort manifest, image, feature cache, or raw study output
is included in the source release. `.gitignore` excludes everything below `data/`
except this index. Treat generated metadata as sensitive too: manifests may contain
patient identifiers, labels, image paths, dates, and hashes.

## Dataset Index

| Dataset / role | Provider and citation | Expected local layout | Code entry point |
|---|---|---|---|
| NIH ChestX-ray14, current study | [NIH Clinical Center collection](https://nihcc.app.box.com/v/ChestXray-NIHCC); cite Wang et al., [ChestX-ray8](https://arxiv.org/abs/1705.02315) | `data/nih_chestxray14/Data_Entry_2017_v2020.csv`, `data/nih_chestxray14/images/`, and local `metadata.json` | `experiments/fetch_nih_probe.py`, then the frozen steps in `EVALUATION_PROTOCOL.md` |
| NIH development and analysis cohorts | Same source and citation as above | `data/nih_chestxray14_replication/`, `data/nih_chestxray14_confirmation/`, and `data/nih_chestxray14_unlabeled_pool/`; each generated manifest and image tree remains local | `experiments/fetch_nih_replication.py`, `experiments/nih_unlabeled_pool.py`, `experiments/fetch_nih_unlabeled_pool.py` |
| NIH sealed confirmation and final holdout | Same source; reserved by the evaluation protocol | `data/nih_chestxray14_sealed_confirmation/` and `data/nih_chestxray14_final_holdout/` | Reserved inputs. Do not download, open, regenerate, or score unless a later approved protocol explicitly changes the status in `docs/handoff/STATUS.md` |
| OpenI / Indiana University chest radiographs | [Open-i, U.S. National Library of Medicine](https://openi.nlm.nih.gov/); use its search/download interface to locate the public chest-radiograph collection | `data/openi_subset/metadata.json` and `data/openi_subset/images/`; metadata records need an `image_filename` field | `experiments/common.py`; legacy quickstart and adapter examples only, not the current scored study. This repository does not include an OpenI bulk downloader or subset manifest. |
| COVID-19 image cohort prototype | [IEEE 8023 COVID-19 Image Data Collection](https://github.com/ieee8023/covid-chestxray-dataset); the downloader pins a source commit and preserves per-image license metadata | `data/covid_cxr2_source_metadata.csv` and `data/covid_cxr2_privacy_{development,confirmation,final_holdout}/` | `experiments/download_patient_cohort.py`; historical prototype, not an input to the current NIH result |

## NIH Download Notes

The NIH source is large and may require accepting provider terms. The current probe
fetcher reads `Data_Entry_2017_v2020.csv` and downloads selected PNG members from a
third-party Hugging Face mirror referenced in `experiments/nih_pneumonia.py`; the
CSV URL is commit-addressed, while the ZIP URL currently uses the mirror's `main`
revision. This is not an immutable archive pin. For a strict reproduction, obtain
the source from NIH, record the archive and metadata SHA-256 values locally, and do
not describe the mirror as the NIH host. The fetcher validates expected archive
sizes and records per-image hashes in the ignored local manifest.

The initial probe command writes only beneath the ignored `data/` directory:

```powershell
.\.venv\Scripts\python.exe experiments\fetch_nih_probe.py
```

Continue only with the exact cohort steps permitted by `EVALUATION_PROTOCOL.md`.
The final confirmation set is intentionally closed. In particular, do not run a
sealed-set fetcher just to make the local tree look complete. A missing reserved
set is a protocol boundary, not a missing repository asset.

On a fresh clone, the provider links above identify where to request or retrieve
the corresponding source data; this repository intentionally does not mirror the
datasets. OpenI quickstart data must be obtained from Open-i and assembled into
the documented local layout; no automatic downloader is provided. For every
locally prepared cohort, preserve the provider/version, access terms, source
revision, and SHA-256 values in a private local manifest. Do not publish that
manifest if it contains patient IDs, image paths, labels, or other restricted
metadata. The NIH probe's CSV is commit-addressed, but its image ZIP URL follows
the mirror's `main` branch, so the checked-in downloader alone is not an immutable
archive pin.

## Reproduction Limits

A clean source clone can install the package and run the synthetic/unit test suite
without any dataset. Reproducing the reported scientific aggregates also requires
lawful access to the exact dataset version, local cohort manifests, the matching
model/checkpoint revisions, and generated feature caches. Those inputs are not
redistributed here, and the reserved confirmation data must remain unopened. See
`docs/ENVIRONMENT.md`, `docs/handoff/STATUS.md`, and `EVALUATION_PROTOCOL.md` before
attempting an experiment.
