"""Download three disjoint repeated-patient privacy cohorts.

The source is the public covid-chestxray-dataset repository at a fixed commit.
Selection is deterministic, sex-stratified, patient-disjoint, and keeps every
eligible X-ray. Development and confirmation use patients with at least three
studies; the immutable final holdout uses previously unused patients with
exactly two studies.
Per-image licenses and source URLs remain in the local metadata manifest.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
import time
import urllib.parse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

COMMIT = "78543292f8b01d5e0ed1d0e15dce71949f0657bb"
REPOSITORY = "ieee8023/covid-chestxray-dataset"
RAW_ROOT = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
SOURCE_METADATA = DATA_ROOT / "covid_cxr2_source_metadata.csv"
DEVELOPMENT_ROOT = DATA_ROOT / "covid_cxr2_privacy_development"
CONFIRMATION_ROOT = DATA_ROOT / "covid_cxr2_privacy_confirmation"
FINAL_HOLDOUT_ROOT = DATA_ROOT / "covid_cxr2_privacy_final_holdout"
MIN_IMAGES_PER_PATIENT = 3
ALLOWED_LICENSES = {
    "Apache 2.0",
    "CC BY",
    "CC BY 3.0",
    "CC BY 4.0",
    "CC BY-SA",
    "CC BY-NC-SA",
    "CC BY-NC-SA 4.0",
}
DOWNLOAD_WORKERS = 8


def _git_blob(repository: Path, relative_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "cat-file", "blob", f"{COMMIT}:{relative_path}"],
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"failed to read {relative_path} from source commit")
    return completed.stdout


def _write_blob(
    repository: Optional[Path],
    relative_path: str,
    target: Path,
    expected_size: Optional[int] = None,
) -> None:
    if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    content = None
    if repository is not None:
        try:
            content = _git_blob(repository, relative_path)
        except (RuntimeError, subprocess.TimeoutExpired):
            content = None
    if content is None:
        url = f"{RAW_ROOT}/{urllib.parse.quote(relative_path)}"
        temporary.unlink(missing_ok=True)
        command = [
            "curl.exe",
            "-L",
            "--fail",
            "--http1.1",
            "--retry",
            "3",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "20",
            "--max-time",
            "90",
            url,
            "-o",
            str(temporary),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"failed to download {url}")
        content = temporary.read_bytes()
    if expected_size is not None and len(content) != expected_size:
        raise IOError(
            f"expected {expected_size} bytes, received {len(content)}"
        )
    temporary.write_bytes(content)
    for attempt in range(5):
        try:
            os.replace(temporary, target)
            break
        except PermissionError:
            # Indexers and virus scanners can briefly hold a newly written file
            # on Windows. A concurrent identical target is already complete.
            if target.is_file() and target.read_bytes() == content:
                temporary.unlink(missing_ok=True)
                break
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))


def _prepare_record(
    record: dict[str, str],
    *,
    source_repository: Optional[Path],
    output_root: Path,
) -> dict[str, str]:
    filename = record["filename"]
    source = f"{RAW_ROOT}/images/{urllib.parse.quote(filename)}"
    target = output_root / "images" / filename
    _write_blob(source_repository, f"images/{filename}", target)
    return {
        **record,
        "relative_path": str(target.relative_to(output_root)).replace("\\", "/"),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source_raw_url": source,
    }


def validate_manifest(value: dict, output_root: Path) -> None:
    """Fail closed unless the local cohort exactly matches the declared design."""
    if value.get("schema_version") != 1 or value.get("commit") != COMMIT:
        raise ValueError("cohort manifest schema or fixed source commit is invalid")
    records = value.get("records")
    selection = value.get("selection")
    if not isinstance(records, list) or not isinstance(selection, dict):
        raise ValueError("cohort manifest records or selection metadata are invalid")
    if len(records) != selection.get("image_count") or len(records) < 100:
        raise ValueError("cohort must contain its declared image count and at least 100 images")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("license") not in ALLOWED_LICENSES:
            raise ValueError("cohort manifest contains a disallowed image license")
        grouped[str(record.get("patient_id", ""))].append(record)
        target = output_root / str(record.get("relative_path", ""))
        if not target.is_file():
            raise FileNotFoundError(f"cohort image is missing: {target}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != record.get("sha256"):
            raise ValueError(f"cohort image digest mismatch: {target}")
    if len(grouped) != selection.get("patient_count") or len(grouped) < 20:
        raise ValueError("cohort must contain its declared patient count and at least 20 patients")
    minimum_images = selection.get("minimum_images_per_patient")
    if minimum_images not in {2, MIN_IMAGES_PER_PATIENT}:
        raise ValueError("cohort minimum images per patient must be two or three")
    if any(len(patient_records) < minimum_images for patient_records in grouped.values()):
        raise ValueError("cohort patient has fewer images than its declared minimum")
    patient_sexes = [records[0].get("sex") for records in grouped.values()]
    expected_sexes = selection.get("patients_by_sex")
    actual_sexes = {sex: patient_sexes.count(sex) for sex in ("F", "M")}
    if actual_sexes != expected_sexes:
        raise ValueError("cohort patient sex counts do not match the manifest")


def validate_disjoint_cohorts(*cohorts: dict) -> None:
    """Reject patient, source-image, or digest overlap across audit cohorts."""
    if len(cohorts) < 2:
        raise ValueError("at least two cohorts are required for overlap validation")
    dimensions = {
        "patient": "patient_id",
        "source image": "filename",
        "image digest": "sha256",
    }
    for label, key in dimensions.items():
        seen: set[str] = set()
        for cohort in cohorts:
            current = {
                str(record.get(key, "")) for record in cohort.get("records", [])
            } - {""}
            overlap = sorted(seen & current)
            if overlap:
                raise ValueError(f"privacy cohorts share {label}: {overlap[:3]}")
            seen.update(current)


def _patient_order(patient_id: str) -> str:
    return hashlib.sha256(f"FHEViT-v181:{patient_id}".encode("utf-8")).hexdigest()


def _select_cohorts(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("modality") == "X-ray"
            and row.get("view", "").startswith(("PA", "AP"))
            and row.get("patientid")
            and row.get("filename")
            and row.get("sex") in {"F", "M"}
            and row.get("license") in ALLOWED_LICENSES
        ):
            grouped[row["patientid"]].append(row)

    selected: dict[str, list[dict[str, str]]] = {
        "development": [],
        "confirmation": [],
        "final_holdout": [],
    }
    for sex in ("F", "M"):
        candidates = [
            (patient_id, records)
            for patient_id, records in grouped.items()
            if len(records) >= MIN_IMAGES_PER_PATIENT and records[0]["sex"] == sex
        ]
        candidates.sort(key=lambda item: _patient_order(item[0]))
        if len(candidates) < 2:
            raise ValueError(f"only {len(candidates)} eligible repeated patients for sex {sex}")
        for patient_index, (patient_id, records) in enumerate(candidates):
            split = "development" if patient_index % 2 == 0 else "confirmation"
            ordered = sorted(records, key=lambda row: (row.get("date", ""), row["filename"]))
            for study_index, row in enumerate(ordered):
                selected[split].append(
                    {
                        "patient_id": patient_id,
                        "study_index": str(study_index),
                        "sex": sex,
                        "age": row.get("age", ""),
                        "finding": row.get("finding", ""),
                        "view": row.get("view", ""),
                        "date": row.get("date", ""),
                        "filename": row["filename"],
                        "license": row["license"],
                        "original_url": row.get("url", ""),
                        "doi": row.get("doi", ""),
                    }
                )
        final_candidates = [
            (patient_id, records)
            for patient_id, records in grouped.items()
            if len(records) == 2 and records[0]["sex"] == sex
        ]
        final_candidates.sort(key=lambda item: _patient_order(item[0]))
        for patient_id, records in final_candidates:
            ordered = sorted(records, key=lambda row: (row.get("date", ""), row["filename"]))
            for study_index, row in enumerate(ordered):
                selected["final_holdout"].append(
                    {
                        "patient_id": patient_id,
                        "study_index": str(study_index),
                        "sex": sex,
                        "age": row.get("age", ""),
                        "finding": row.get("finding", ""),
                        "view": row.get("view", ""),
                        "date": row.get("date", ""),
                        "filename": row["filename"],
                        "license": row["license"],
                        "original_url": row.get("url", ""),
                        "doi": row.get("doi", ""),
                    }
                )
    return selected


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if SOURCE_METADATA.exists() and SOURCE_METADATA.stat().st_size == 599143:
        source_repository = None
        _build_cohorts_from_metadata(source_repository)
        return
    with tempfile.TemporaryDirectory(prefix="fhevit_v181_source_") as temporary:
        repository = Path(temporary) / "source"
        fetch_result: Optional[subprocess.CompletedProcess[bytes]] = None
        repository.mkdir()
        subprocess.run(["git", "init", "--quiet"], cwd=repository, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", f"https://github.com/{REPOSITORY}.git"],
            cwd=repository,
            check=True,
        )
        fetch_command = [
            "git",
            "-c",
            "http.version=HTTP/1.1",
            "fetch",
            "--depth",
            "1",
            "--filter=blob:none",
            "origin",
            COMMIT,
        ]
        for attempt in range(3):
            try:
                fetch_result = subprocess.run(
                    fetch_command,
                    cwd=repository,
                    check=False,
                    timeout=180,
                )
            except subprocess.TimeoutExpired:
                fetch_result = None
            if fetch_result is not None and fetch_result.returncode == 0:
                break
            if attempt < 2:
                time.sleep(2**attempt)
        source_repository = (
            repository
            if fetch_result is not None and fetch_result.returncode == 0
            else None
        )
        if source_repository is not None:
            actual_commit = subprocess.run(
                ["git", "rev-parse", "FETCH_HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if actual_commit != COMMIT:
                raise RuntimeError(f"source HEAD changed: expected {COMMIT}, got {actual_commit}")
        metadata_csv = SOURCE_METADATA
        if not metadata_csv.exists() or metadata_csv.stat().st_size != 599143:
            _write_blob(source_repository, "metadata.csv", metadata_csv, expected_size=599143)
        with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        cohorts = _select_cohorts(rows)
        manifests = {}
        for split, selected in cohorts.items():
            roots = {
                "development": DEVELOPMENT_ROOT,
                "confirmation": CONFIRMATION_ROOT,
                "final_holdout": FINAL_HOLDOUT_ROOT,
            }
            output_root = roots[split]
            manifest: list[Optional[dict[str, str]]] = [None] * len(selected)
            with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
                futures = {
                    executor.submit(
                        _prepare_record,
                        record,
                        source_repository=source_repository,
                        output_root=output_root,
                    ): index
                    for index, record in enumerate(selected)
                }
                for completed_count, future in enumerate(as_completed(futures), start=1):
                    manifest[futures[future]] = future.result()
                    if completed_count % 20 == 0:
                        print(
                            f"Prepared {split} {completed_count}/{len(selected)}",
                            flush=True,
                        )
            if any(record is None for record in manifest):
                raise RuntimeError(f"incomplete record preparation for {split}")
            complete_manifest = [record for record in manifest if record is not None]
            patients_by_sex = {
                sex: len(
                    {
                        record["patient_id"]
                        for record in complete_manifest
                        if record["sex"] == sex
                    }
                )
                for sex in ("F", "M")
            }
            output = {
                "schema_version": 1,
                "repository": REPOSITORY,
                "commit": COMMIT,
                "split": split,
                "selection": {
                    "minimum_images_per_patient": (
                        2 if split == "final_holdout" else MIN_IMAGES_PER_PATIENT
                    ),
                    "patient_count": sum(patients_by_sex.values()),
                    "patients_by_sex": patients_by_sex,
                    "image_count": len(complete_manifest),
                    "allowed_licenses": sorted(ALLOWED_LICENSES),
                },
                "records": complete_manifest,
            }
            metadata_json = output_root / "metadata.json"
            metadata_json.write_text(
                json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            validate_manifest(output, output_root)
            manifests[split] = output
            print(f"Saved {len(complete_manifest)} images for {split}: {metadata_json}")
        validate_disjoint_cohorts(
            manifests["development"],
            manifests["confirmation"],
            manifests["final_holdout"],
        )


def _build_cohorts_from_metadata(source_repository: Optional[Path]) -> None:
    metadata_csv = SOURCE_METADATA
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cohorts = _select_cohorts(rows)
    manifests = {}
    for split, selected in cohorts.items():
        roots = {
            "development": DEVELOPMENT_ROOT,
            "confirmation": CONFIRMATION_ROOT,
            "final_holdout": FINAL_HOLDOUT_ROOT,
        }
        output_root = roots[split]
        manifest: list[Optional[dict[str, str]]] = [None] * len(selected)
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            futures = {
                executor.submit(
                    _prepare_record,
                    record,
                    source_repository=source_repository,
                    output_root=output_root,
                ): index
                for index, record in enumerate(selected)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                manifest[futures[future]] = future.result()
                if completed % 20 == 0:
                    print(f"Prepared {split} {completed}/{len(selected)}", flush=True)
        if any(record is None for record in manifest):
            raise RuntimeError(f"incomplete record preparation for {split}")
        complete_manifest = [record for record in manifest if record is not None]
        patients_by_sex = {
            sex: len({record["patient_id"] for record in complete_manifest if record["sex"] == sex})
            for sex in ("F", "M")
        }
        output = {
            "schema_version": 1,
            "repository": REPOSITORY,
            "commit": COMMIT,
            "split": split,
            "selection": {
                "minimum_images_per_patient": (
                    2 if split == "final_holdout" else MIN_IMAGES_PER_PATIENT
                ),
                "patient_count": sum(patients_by_sex.values()),
                "patients_by_sex": patients_by_sex,
                "image_count": len(complete_manifest),
                "allowed_licenses": sorted(ALLOWED_LICENSES),
            },
            "records": complete_manifest,
        }
        metadata_json = output_root / "metadata.json"
        metadata_json.write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        validate_manifest(output, output_root)
        manifests[split] = output
        print(f"Saved {len(complete_manifest)} images for {split}: {metadata_json}")
    validate_disjoint_cohorts(
        manifests["development"],
        manifests["confirmation"],
        manifests["final_holdout"],
    )


if __name__ == "__main__":
    main()
