"""Count the section 2.22 image-level pool. Do not relabel sealed manifests.

A sealed confirmation manifest is written only when both classes have at least
253 remaining patients. Images are not downloaded here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.nih_pneumonia import (  # noqa: E402
    EXPECTED_ROWS,
    classify_image_level,
    patient_order,
    rows_from_csv,
)

PER_CLASS = 253
CSV = ROOT / "data" / "nih_chestxray14" / "Data_Entry_2017_v2020.csv"
MANIFESTS = (
    ROOT / "data" / "nih_chestxray14" / "metadata.json",
    ROOT / "data" / "nih_chestxray14_replication" / "metadata.json",
    ROOT / "data" / "nih_chestxray14_confirmation" / "metadata.json",
)
SEALED = ROOT / "data" / "nih_chestxray14_sealed_confirmation" / "metadata.json"


def gate_open(pneumonia, no_finding, per_class=PER_CLASS):
    """Both classes must cover the section 2.19 zero-difference plan."""
    if per_class != PER_CLASS:
        raise ValueError("the locked class size is 253")
    return pneumonia >= per_class and no_finding >= per_class


def image_level_pools(rows, exclude_ids):
    excluded = {str(patient) for patient in exclude_ids}
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["patient_id"]).strip(), []).append(row)
    pools = {0: [], 1: []}
    for patient, records in grouped.items():
        if patient in excluded:
            continue
        decision = classify_image_level(records)
        if decision is None:
            continue
        label, row = decision
        pools[label].append({
            "patient_id": patient,
            "image_index": row["image_index"],
            "finding_labels": row["finding_labels"],
            "follow_up": int(row["follow_up"]),
            "patient_age": int(str(row["patient_age"]).strip()),
            "patient_gender": (row.get("patient_gender") or "").strip(),
            "view_position": row["view_position"].strip(),
            "label_pneumonia": label,
        })
    return pools


def _excluded():
    ids = []
    for path in MANIFESTS:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        ids.extend(str(row["patient_id"]) for row in manifest["records"])
    if len(ids) != len(set(ids)) or len(ids) != 171:
        raise SystemExit("the three sealed manifests are not 171 distinct patients")
    return ids


def _record(row):
    return {
        **row,
        "relative_path": f"images/{row['image_index']}",
        "sha256": None,
        "download_error": "not_fetched",
    }


def main():
    rows = rows_from_csv(CSV.read_text(encoding="utf-8"))
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit("the local NIH table is not the expected 112120 rows")
    excluded = _excluded()
    pools = image_level_pools(rows, excluded)
    pneumonia = sorted(pools[1], key=lambda row: patient_order(row["patient_id"]))
    negative = sorted(pools[0], key=lambda row: patient_order(row["patient_id"]))
    opened = gate_open(len(pneumonia), len(negative))
    result = {
        "schema": "nih_image_level_census_v1",
        "protocol_section": "2.22",
        "per_class": PER_CLASS,
        "excluded_patients": len(excluded),
        "remaining_pneumonia": len(pneumonia),
        "remaining_no_finding": len(negative),
        "gate_open": opened,
        "margin": 0.05,
        "primary_rank": 4,
        "secondary_rank": 8,
        "sealed_manifests_rewritten": False,
        "images_downloaded": False,
        "auc_computed": False,
        "predictor_retarget_started": False,
        "mlp_started": False,
        "privacy_channel_redesign_started": False,
        "regional_release_started": False,
        "medgemma_ran": False,
        "chexpert_downloaded": False,
        "mimic_downloaded": False,
        "direction_nulling_started": False,
        "reconstruction_attack_started": False,
    }
    if opened:
        chosen = [_record(row) for row in pneumonia[:PER_CLASS] + negative[:PER_CLASS]]
        if len(chosen) != 2 * PER_CLASS or len({row["patient_id"] for row in chosen}) != len(chosen):
            raise SystemExit("the sealed list is not two disjoint classes of 253")
        if set(row["patient_id"] for row in chosen) & set(excluded):
            raise SystemExit("a sealed patient entered the new list")
        SEALED.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": "nih_sealed_confirmation_v1",
            "protocol_section": "2.22",
            "source": "NIH ChestX-ray14",
            "label_file": "Data_Entry_2017_v2020.csv",
            "eligibility": "image_level_pneumonia_patient_level_no_finding",
            "per_class": PER_CLASS,
            "primary_rank": 4,
            "secondary_rank": 8,
            "margin": 0.05,
            "pool_complete": False,
            "auc_computed": False,
            "official_test_list_read": False,
            "records": chosen,
        }
        SEALED.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        result["manifest_written"] = True
        result["listed_patients"] = len(chosen)
    else:
        result["manifest_written"] = False
        result["listed_patients"] = 0
    destination = ROOT / "experiments" / "results" / "nih_image_level_census.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "remaining_pneumonia": len(pneumonia),
        "remaining_no_finding": len(negative),
        "gate_open": opened,
        "manifest_written": result["manifest_written"],
        "images_downloaded": False,
        "auc_computed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
