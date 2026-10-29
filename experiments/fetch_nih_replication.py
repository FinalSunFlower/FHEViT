"""Download the locked NIH replication, excluding the section 2.5 patients.

The slice is EVALUATION_PROTOCOL.md section 2.6. A shortfall is not repaired.
A failed member is not replaced by a later patient. The section 2.5 manifest
is read and not rewritten.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fetch_nih_probe import download_csv, fetch_range, locate  # noqa: E402
from experiments.nih_pneumonia import (  # noqa: E402
    NEGATIVE_LIMIT,
    POSITIVE_LIMIT,
    REPLICATION_NEGATIVE,
    REPLICATION_POSITIVE,
    ZIP_BYTES,
    ZIP_URL,
    select_probe,
)
from experiments.zip_member import extract_member  # noqa: E402

SEALED = ROOT / "data" / "nih_chestxray14" / "metadata.json"
DESTINATION = ROOT / "data" / "nih_chestxray14_replication"


def sealed_ids():
    manifest = json.loads(SEALED.read_text(encoding="utf-8"))
    records = manifest.get("records") or []
    if not manifest.get("pool_complete") or len(records) != POSITIVE_LIMIT + NEGATIVE_LIMIT:
        raise ValueError("section 2.5 manifest is not a complete 27-patient probe")
    if any(row.get("download_error") for row in records):
        raise ValueError("section 2.5 manifest still has a failed download")
    return {str(row["patient_id"]) for row in records}


def main():
    excluded = sealed_ids()
    rows, csv_sha = download_csv()
    positives, negatives, enough = select_probe(
        rows,
        positive_limit=REPLICATION_POSITIVE,
        negative_limit=REPLICATION_NEGATIVE,
        exclude_ids=excluded,
    )
    chosen = positives + negatives
    if {row["patient_id"] for row in chosen} & excluded:
        raise ValueError("replication selected a sealed patient")
    images = DESTINATION / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest_path = DESTINATION / "metadata.json"
    if not enough:
        payload = {
            "schema": "nih_replication_probe_v1",
            "shortfall": True,
            "eligible_pneumonia": len(positives),
            "eligible_no_finding": len(negatives),
            "positive_limit": REPLICATION_POSITIVE,
            "negative_limit": REPLICATION_NEGATIVE,
            "excluded_patients": len(excluded),
            "auc_computed": False,
            "csv_sha256": csv_sha,
        }
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"shortfall": True, "eligible_pneumonia": len(positives),
                          "eligible_no_finding": len(negatives)}))
        return
    located = locate([row["image_index"] for row in chosen])
    records = []
    for row in chosen:
        member = located.get(row["image_index"])
        target = images / row["image_index"]
        error = None
        digest = None
        if member is None:
            error = "image is not in the pinned archives"
        else:
            try:
                if not target.is_file():
                    url = ZIP_URL.format(member["zip"])
                    raw = extract_member(
                        lambda start, end, url=url, size=ZIP_BYTES[member["zip"]]: fetch_range(url, start, end, size),
                        member,
                    )
                    if not raw.startswith(b"\x89PNG"):
                        raise ValueError("member is not a PNG")
                    target.write_bytes(raw)
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                if not target.read_bytes().startswith(b"\x89PNG"):
                    raise ValueError("saved file is not a PNG")
            except Exception as exc:  # noqa: BLE001 - recorded on this patient, not replaced
                error = str(exc)
        records.append({**row, "relative_path": f"images/{row['image_index']}", "sha256": digest,
                        "download_error": error})
        print(json.dumps({"patient_id": row["patient_id"], "image": row["image_index"], "error": error}), flush=True)
    complete = all(row["download_error"] is None for row in records)
    payload = {
        "schema": "nih_replication_probe_v1",
        "source": "NIH ChestX-ray14",
        "citation": "Wang et al., ChestX-ray8, CVPR 2017",
        "source_site": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "label_file": "Data_Entry_2017_v2020.csv",
        "csv_sha256": csv_sha,
        "excluded_section": "2.5",
        "excluded_patients": len(excluded),
        "shortfall": False,
        "pool_complete": complete,
        "pneumonia_patients": REPLICATION_POSITIVE,
        "no_finding_patients": REPLICATION_NEGATIVE,
        "auc_computed": False,
        "official_test_list_read": False,
        "pca_refit": False,
        "bridge_refit": False,
        "medgemma_ran": False,
        "records": records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pool_complete": complete, "patients": len(records)}))


if __name__ == "__main__":
    main()
