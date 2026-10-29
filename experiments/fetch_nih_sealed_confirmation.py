"""Download the images named by the section 2.22 manifest.

The patient list is fixed. A missing or failed image is recorded and is not
replaced. Scoring stays off until every image is stored.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fetch_nih_probe import fetch_range, locate  # noqa: E402
from experiments.nih_image_level_census import PER_CLASS  # noqa: E402
from experiments.nih_pneumonia import ZIP_BYTES, ZIP_URL  # noqa: E402
from experiments.zip_member import extract_member  # noqa: E402

MANIFEST = ROOT / "data" / "nih_chestxray14_sealed_confirmation" / "metadata.json"


def main():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = payload["records"]
    if payload.get("protocol_section") != "2.22" or len(records) != 2 * PER_CLASS:
        raise SystemExit("the sealed manifest is not the locked 253 + 253 list")
    if sum(row["label_pneumonia"] == 1 for row in records) != PER_CLASS:
        raise SystemExit("the sealed list is not balanced")
    images = MANIFEST.parent / "images"
    images.mkdir(parents=True, exist_ok=True)
    pending = [row["image_index"] for row in records if not (images / row["image_index"]).is_file()]
    located = locate(pending) if pending else {}
    for row in records:
        target = images / row["image_index"]
        error = None
        digest = row.get("sha256")
        try:
            if not target.is_file():
                member = located.get(row["image_index"])
                if member is None:
                    raise ValueError("image is not in the pinned archives")
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
            digest = None
        row["relative_path"] = f"images/{row['image_index']}"
        row["sha256"] = digest
        row["download_error"] = error
        print(json.dumps({"patient_id": row["patient_id"], "image": row["image_index"], "error": error}), flush=True)
    payload["records"] = records
    payload["pool_complete"] = all(row["download_error"] is None for row in records)
    payload["auc_computed"] = False
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pool_complete": payload["pool_complete"], "patients": len(records)}))


if __name__ == "__main__":
    main()
