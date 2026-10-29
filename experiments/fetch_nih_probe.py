"""Download the locked NIH ChestX-ray14 probe.

Eligibility is decided from Data_Entry_2017_v2020.csv. Image bytes are fetched
after that list is fixed. A failed member is recorded and is not replaced by a
later patient. The official test list is not read.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.nih_pneumonia import (  # noqa: E402
    CSV_URL,
    EXPECTED_ROWS,
    NEGATIVE_LIMIT,
    POSITIVE_LIMIT,
    ZIP_BYTES,
    ZIP_URL,
    rows_from_csv,
    select_probe,
)
from experiments.zip_member import extract_member, read_central_directory  # noqa: E402

DESTINATION = ROOT / "data" / "nih_chestxray14"
USER_AGENT = "FHEViT-nih-probe/1.0"


def fetch_range(url, start, end, expected_size):
    last_error = None
    for _ in range(3):
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
                if total != expected_size:
                    raise ValueError(f"{url} size {total} does not match the pinned archive")
                body = response.read()
            if len(body) != end - start + 1:
                raise ValueError(f"range {start}-{end} returned {len(body)} bytes")
            return body
        except Exception as exc:  # noqa: BLE001 - the same range is retried, the patient list is not
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"range {start}-{end} failed for {url}") from last_error


def download_csv():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    target = DESTINATION / "Data_Entry_2017_v2020.csv"
    if not target.is_file():
        request = urllib.request.Request(CSV_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            target.write_bytes(response.read())
    text = target.read_text(encoding="utf-8")
    rows = rows_from_csv(text)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} NIH rows, found {len(rows)}")
    return rows, hashlib.sha256(target.read_bytes()).hexdigest()


def locate(image_names):
    found = {}
    remaining = set(image_names)
    for number, size in ZIP_BYTES.items():
        if not remaining:
            break
        url = ZIP_URL.format(number)

        def fetch(start, end, url=url, size=size):
            return fetch_range(url, start, end, size)

        for name, member in read_central_directory(fetch, size).items():
            base = name.replace("\\", "/").rsplit("/", 1)[-1]
            if base in remaining:
                found[base] = {"zip": number, **member}
                remaining.remove(base)
    return found


def main():
    rows, csv_sha = download_csv()
    positives, negatives, enough = select_probe(rows)
    chosen = positives + negatives
    images = DESTINATION / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest_path = DESTINATION / "metadata.json"
    if not enough:
        payload = {
            "schema": "nih_pneumonia_probe_v1",
            "shortfall": True,
            "eligible_pneumonia": len(positives),
            "eligible_no_finding": len(negatives),
            "positive_limit": POSITIVE_LIMIT,
            "negative_limit": NEGATIVE_LIMIT,
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
        "schema": "nih_pneumonia_probe_v1",
        "source": "NIH ChestX-ray14",
        "citation": "Wang et al., ChestX-ray8, CVPR 2017",
        "source_site": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "label_file": "Data_Entry_2017_v2020.csv",
        "csv_sha256": csv_sha,
        "shortfall": False,
        "pool_complete": complete,
        "pneumonia_patients": POSITIVE_LIMIT,
        "no_finding_patients": NEGATIVE_LIMIT,
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
