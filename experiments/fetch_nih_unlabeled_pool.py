"""Download the frozen section 2.27 unlabeled prefix. Failures are not replaced.

The patient list is already written. This script does not add patients,
does not read finding labels, and does not score an AUC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fetch_nih_probe import fetch_range, locate  # noqa: E402
from experiments.nih_pneumonia import ZIP_BYTES, ZIP_URL  # noqa: E402
from experiments.zip_member import extract_member  # noqa: E402

MANIFEST = ROOT / "data" / "nih_chestxray14_unlabeled_pool" / "metadata.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8000)
    args = parser.parse_args()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = payload["records"]
    if payload.get("protocol_section") != "2.27" or not payload.get("schedule_feasible"):
        raise SystemExit("the unlabeled prefix is not the locked schedule")
    if len(records) != 8000 or payload.get("confirmation_opened") or payload.get("auc_computed"):
        raise SystemExit("the unlabeled list is not a frozen unscored prefix of 8000")
    if not 1 <= args.limit <= len(records):
        raise SystemExit("the download limit must stay inside the frozen 8000")
    wanted = records[:args.limit]
    images = MANIFEST.parent / "images"
    images.mkdir(parents=True, exist_ok=True)
    pending = [row["image_index"] for row in wanted if not (images / row["image_index"]).is_file()]
    located = locate(pending) if pending else {}
    for row in wanted:
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
            saved = target.read_bytes()
            if not saved.startswith(b"\x89PNG"):
                raise ValueError("saved file is not a PNG")
            digest = hashlib.sha256(saved).hexdigest()
        except Exception as exc:  # noqa: BLE001 - recorded on this patient, not replaced
            error = str(exc)
            digest = None
        row["relative_path"] = f"images/{row['image_index']}"
        row["sha256"] = digest
        row["download_error"] = error
        print(json.dumps({"patient_id": row["patient_id"], "image": row["image_index"], "error": error}), flush=True)
    payload["records"] = records
    payload["download_prefix"] = args.limit
    payload["download_prefix_complete"] = all(row.get("download_error") is None for row in wanted)
    payload["images_downloaded"] = args.limit == len(records) and payload["download_prefix_complete"]
    payload["features_cached"] = False
    payload["auc_computed"] = False
    payload["attacker_scored"] = False
    payload["confirmation_opened"] = False
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "download_prefix": args.limit,
        "download_prefix_complete": payload["download_prefix_complete"],
        "images_downloaded": payload["images_downloaded"],
    }), flush=True)


if __name__ == "__main__":
    main()
