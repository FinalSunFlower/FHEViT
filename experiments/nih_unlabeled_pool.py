"""List unlabeled NIH patients for the bridge curve. No image is downloaded.

EVALUATION_PROTOCOL.md section 2.27. Finding labels are not a selection
rule. The reserved confirmation patients stay out of the list.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.nih_pneumonia import (  # noqa: E402
    EXPECTED_ROWS,
    _chosen,
    patient_order,
    rows_from_csv,
)

PREFIXES = (500, 2000, 8000)
CSV = ROOT / "data" / "nih_chestxray14" / "Data_Entry_2017_v2020.csv"
HOLDOUT = ROOT / "data" / "nih_chestxray14_final_holdout" / "positives.json"
MANIFESTS = (
    ROOT / "data" / "nih_chestxray14" / "metadata.json",
    ROOT / "data" / "nih_chestxray14_replication" / "metadata.json",
    ROOT / "data" / "nih_chestxray14_confirmation" / "metadata.json",
    ROOT / "data" / "nih_chestxray14_sealed_confirmation" / "metadata.json",
)
DESTINATION = ROOT / "data" / "nih_chestxray14_unlabeled_pool" / "metadata.json"


def choose_unlabeled_film(records):
    """Earliest PA or AP film. Age is judged on that film. A later film is not promoted."""
    return _chosen(records, lambda item: (item.get("view_position") or "").strip() in {"PA", "AP"})


def ordered_patients(rows, exclude_ids):
    excluded = {str(patient) for patient in exclude_ids}
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["patient_id"]).strip(), []).append(row)
    chosen = []
    for patient, records in grouped.items():
        if patient in excluded:
            continue
        film = choose_unlabeled_film(records)
        if film is None:
            continue
        chosen.append(film)
    chosen.sort(key=lambda row: patient_order(str(row["patient_id"]).strip()))
    return chosen


def main():
    rows = rows_from_csv(CSV.read_text(encoding="utf-8"))
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit("the NIH table is not the locked row count")
    excluded = set()
    for path in MANIFESTS:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(str(row["patient_id"]) for row in manifest["records"])
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    if holdout.get("images_downloaded") or holdout.get("auc_computed"):
        raise SystemExit("the reserved confirmation pool was already opened")
    excluded.update(str(row["patient_id"]) for row in holdout["records"])
    ordered = ordered_patients(rows, excluded)
    feasible = len(ordered) >= PREFIXES[-1]
    prefix = ordered[:PREFIXES[-1]] if feasible else []
    records = [
        {
            "patient_id": str(row["patient_id"]).strip(),
            "image_index": row["image_index"],
            "follow_up": int(row["follow_up"]),
            "patient_age": int(str(row["patient_age"]).strip()),
            "patient_gender": (row.get("patient_gender") or "").strip(),
            "view_position": (row["view_position"] or "").strip(),
        }
        for row in prefix
    ]
    digest = hashlib.sha256("\n".join(row["patient_id"] for row in records).encode("utf-8")).hexdigest()
    payload = {
        "schema": "nih_unlabeled_pool_v1",
        "protocol_section": "2.27",
        "eligible_patients": len(ordered),
        "excluded_patients": len(excluded),
        "prefixes": list(PREFIXES),
        "schedule_feasible": feasible,
        "patient_count": len(records),
        "patient_id_sha256": digest,
        "images_downloaded": False,
        "features_cached": False,
        "auc_computed": False,
        "attacker_scored": False,
        "confirmation_opened": False,
        "finding_labels_used_for_selection": False,
        "records": records,
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "eligible_patients": len(ordered),
        "schedule_feasible": feasible,
        "patient_count": len(records),
    }))


if __name__ == "__main__":
    main()
