"""Pneumonia versus No Finding on NIH ChestX-ray14.

The rule matches EVALUATION_PROTOCOL.md section 2.5. A shortfall is not repaired
by promoting a later image, dropping a comorbid token, or shrinking the probe.
"""
from __future__ import annotations

import csv
import hashlib
import io

COVID_PROMPT = "Does this chest X-ray show pneumonia? Answer Yes or No."
PROMPT = COVID_PROMPT
POSITIVE_LIMIT = 14
NEGATIVE_LIMIT = 13
REPLICATION_POSITIVE = 36
REPLICATION_NEGATIVE = 36
CONFIRMATION_POSITIVE = REPLICATION_POSITIVE
CONFIRMATION_NEGATIVE = REPLICATION_NEGATIVE
EXPECTED_ROWS = 112120
AGE_LOW = 18
AGE_HIGH = 90
ORDER_SALT = "FHEViT-nih14-v1:"

CSV_URL = (
    "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/"
    "73b1f1ac6d5b14b4d59dac8ee17a42d69baa47b4/data/Data_Entry_2017_v2020.csv"
)
ZIP_URL = (
    "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data/images/images_{:03d}.zip"
)
ZIP_BYTES = {
    1: 2010553014,
    2: 3952086082,
    3: 3933374676,
    4: 3843097813,
    5: 3939677731,
    6: 3985815542,
    7: 4015817466,
    8: 4017866318,
    9: 4110841707,
    10: 4181069714,
    11: 4186590070,
    12: 2913827342,
}
HEADER = (
    "Image Index",
    "Finding Labels",
    "Follow-up #",
    "Patient ID",
    "Patient Age",
    "Patient Gender",
    "View Position",
)


def patient_order(patient_id):
    return hashlib.sha256(f"{ORDER_SALT}{patient_id}".encode("utf-8")).hexdigest()


def label_tokens(text):
    return tuple(part.strip() for part in (text or "").split("|") if part.strip())


def _age(value):
    text = (value or "").strip()
    if not text.isdigit():
        return None
    return int(text)


def _rows_by_patient(rows):
    grouped, order = {}, []
    for row in rows:
        patient = str(row["patient_id"]).strip()
        if patient not in grouped:
            grouped[patient] = []
            order.append(patient)
        grouped[patient].append(row)
    return grouped, order


def _chosen(records, predicate):
    films = [row for row in records if predicate(row)]
    films.sort(key=lambda row: (int(row["follow_up"]), row["image_index"]))
    if not films:
        return None
    row = films[0]
    age = _age(row["patient_age"])
    view = (row["view_position"] or "").strip()
    if age is None or not (AGE_LOW <= age <= AGE_HIGH) or view not in {"PA", "AP"}:
        return None
    return row


def classify_patient(records):
    """Return (label, row) or None. The earliest failing row is not replaced."""
    parsed = [(row, label_tokens(row["finding_labels"])) for row in records]
    no_finding = [row for row, tokens in parsed if tokens == ("No Finding",)]
    pneumonia = [row for row, tokens in parsed if "Pneumonia" in tokens]
    if no_finding and pneumonia:
        return None
    if len(no_finding) == len(records) and records:
        row = _chosen(records, lambda item: (item["view_position"] or "").strip() in {"PA", "AP"})
        if row is None:
            return None
        return 0, row
    if pneumonia:
        row = _chosen(pneumonia, lambda item: (item["view_position"] or "").strip() in {"PA", "AP"})
        if row is None:
            return None
        return 1, row
    return None


def classify_image_level(records):
    """Section 2.22. Negatives stay patient-level. Positives are one pneumonia image.

    A patient with both a No Finding row and a pneumonia image is positive when
    the earliest valid pneumonia image passes. That patient is never negative.
    A later image is not promoted. This function does not relabel a sealed manifest.
    """
    parsed = [(row, label_tokens(row["finding_labels"])) for row in records]
    no_finding = [row for row, tokens in parsed if tokens == ("No Finding",)]
    pneumonia = [row for row, tokens in parsed if "Pneumonia" in tokens]
    if len(no_finding) == len(records) and records:
        row = _chosen(records, lambda item: (item["view_position"] or "").strip() in {"PA", "AP"})
        if row is None:
            return None
        return 0, row
    if pneumonia:
        row = _chosen(pneumonia, lambda item: (item["view_position"] or "").strip() in {"PA", "AP"})
        if row is None:
            return None
        return 1, row
    return None


def select_probe(rows, *, positive_limit=POSITIVE_LIMIT, negative_limit=NEGATIVE_LIMIT, exclude_ids=()):
    excluded = {str(patient) for patient in exclude_ids}
    grouped, order = _rows_by_patient(rows)
    pools = {0: [], 1: []}
    for patient in order:
        if patient in excluded:
            continue
        decision = classify_patient(grouped[patient])
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
    enough = len(pools[1]) >= positive_limit and len(pools[0]) >= negative_limit
    if not enough:
        return pools[1], pools[0], False
    positives = sorted(pools[1], key=lambda row: patient_order(row["patient_id"]))[:positive_limit]
    negatives = sorted(pools[0], key=lambda row: patient_order(row["patient_id"]))[:negative_limit]
    return positives, negatives, True


def rows_from_csv(text):
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    index = [header.index(name) for name in HEADER]
    rows = []
    for record in reader:
        if not record or not any(cell.strip() for cell in record):
            continue
        image, finding, follow, patient, age, gender, view = (record[item].strip() for item in index)
        rows.append({
            "image_index": image,
            "finding_labels": finding,
            "follow_up": follow,
            "patient_id": patient,
            "patient_age": age,
            "patient_gender": gender,
            "view_position": view,
        })
    return rows
