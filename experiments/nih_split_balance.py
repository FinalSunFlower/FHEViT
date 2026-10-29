"""Do the hash-ordered NIH slices match on age, sex, view, and subtype?

EVALUATION_PROTOCOL.md section 2.11. The two manifests are fixed. The null is
a random 36/36 partition. A stratified split is not started here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.nih_pneumonia import label_tokens  # noqa: E402

DRAWS = 2000
SEED = 42
THRESHOLD = 0.05 / 4
TAKE = 36
COMORBID = (
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
)
PRIMARY = (
    "age_mean_difference",
    "female_proportion_difference",
    "pa_proportion_difference",
    "subtype_l1",
)


def _age(row):
    value = row["patient_age"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError("patient age must be an integer year")
    return int(value)


def _check_row(row):
    if row["patient_gender"] not in {"F", "M"}:
        raise ValueError("patient gender must be F or M")
    if row["view_position"] not in {"PA", "AP"}:
        raise ValueError("view must be PA or AP")
    if row["label_pneumonia"] not in {0, 1}:
        raise ValueError("pneumonia label must be binary")
    _age(row)
    return label_tokens(row["finding_labels"])


def subtype_rates(rows):
    pneumonia = [row for row in rows if row["label_pneumonia"] == 1]
    if len(pneumonia) != TAKE:
        raise ValueError("subtype rates are defined for 36 pneumonia patients")
    parsed = [label_tokens(row["finding_labels"]) for row in pneumonia]
    bare = sum(tokens == ("Pneumonia",) for tokens in parsed) / TAKE
    rates = [bare]
    for name in COMORBID:
        rates.append(sum(name in tokens for tokens in parsed) / TAKE)
    return np.asarray(rates, dtype=float)


def statistics(development, confirmation):
    """Development minus confirmation. Subtype distance is an L1, so it is nonnegative."""
    dev_age = np.asarray([_age(row) for row in development], dtype=float)
    conf_age = np.asarray([_age(row) for row in confirmation], dtype=float)
    dev_female = np.mean([row["patient_gender"] == "F" for row in development])
    conf_female = np.mean([row["patient_gender"] == "F" for row in confirmation])
    dev_pa = np.mean([row["view_position"] == "PA" for row in development])
    conf_pa = np.mean([row["view_position"] == "PA" for row in confirmation])
    return {
        "age_mean_difference": float(dev_age.mean() - conf_age.mean()),
        "female_proportion_difference": float(dev_female - conf_female),
        "pa_proportion_difference": float(dev_pa - conf_pa),
        "subtype_l1": float(np.abs(subtype_rates(development) - subtype_rates(confirmation)).sum()),
    }


def _prepare(development, confirmation):
    for row in (*development, *confirmation):
        _check_row(row)
    if len(development) != 2 * TAKE or len(confirmation) != 2 * TAKE:
        raise ValueError("each slice must contain 72 patients")
    for rows in (development, confirmation):
        if sum(row["label_pneumonia"] == 1 for row in rows) != TAKE:
            raise ValueError("each slice must contain 36 pneumonia patients")
    dev_ids = [row["patient_id"] for row in development]
    conf_ids = [row["patient_id"] for row in confirmation]
    if len(set(dev_ids) | set(conf_ids)) != 4 * TAKE:
        raise ValueError("the two slices must be disjoint and contain unique patients")
    known = {"Pneumonia", "No Finding", *COMORBID}
    unexpected = sorted({
        token for row in (*development, *confirmation) for token in label_tokens(row["finding_labels"])
        if token not in known
    })
    return unexpected


def assess(development, confirmation, *, draws=DRAWS, seed=SEED):
    unexpected = _prepare(development, confirmation)
    observed = statistics(development, confirmation)
    pneumonia = [row for row in (*development, *confirmation) if row["label_pneumonia"] == 1]
    negative = [row for row in (*development, *confirmation) if row["label_pneumonia"] == 0]
    rng = np.random.default_rng(seed)
    exceed = {key: 0 for key in PRIMARY}
    for _ in range(draws):
        take_pos = rng.choice(len(pneumonia), size=TAKE, replace=False)
        take_neg = rng.choice(len(negative), size=TAKE, replace=False)
        chosen = set(take_pos.tolist()) | {len(pneumonia) + i for i in take_neg.tolist()}
        pool = pneumonia + negative
        left = [row for i, row in enumerate(pool) if i in chosen]
        right = [row for i, row in enumerate(pool) if i not in chosen]
        sample = statistics(left, right)
        for key in PRIMARY:
            if abs(sample[key]) >= abs(observed[key]):
                exceed[key] += 1
    p_values = {key: (1 + exceed[key]) / (draws + 1) for key in PRIMARY}
    flagged = [key for key in PRIMARY if p_values[key] < THRESHOLD]
    dev_rates = subtype_rates(development)
    conf_rates = subtype_rates(confirmation)
    names = ["bare_Pneumonia", *COMORBID]
    return {
        "observed": observed,
        "p_values": p_values,
        "threshold": THRESHOLD,
        "imbalanced": bool(flagged),
        "flagged": flagged,
        "unexpected_tokens": unexpected,
        "subtype_rates": {
            name: {"development": float(dev_rates[i]), "confirmation": float(conf_rates[i])}
            for i, name in enumerate(names)
        },
        "draws": draws,
        "seed": seed,
    }


def main():
    development = json.loads(
        (ROOT / "data" / "nih_chestxray14_replication" / "metadata.json").read_text(encoding="utf-8"))
    confirmation = json.loads(
        (ROOT / "data" / "nih_chestxray14_confirmation" / "metadata.json").read_text(encoding="utf-8"))
    result = assess(development["records"], confirmation["records"])
    payload = {
        "schema": "nih_split_balance_v1",
        "protocol_section": "2.11",
        "development_patients": 72,
        "confirmation_patients": 72,
        "stratified_split_started": False,
        "patients_added": False,
        **result,
    }
    destination = ROOT / "experiments" / "results" / "nih_split_balance.json"
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "imbalanced": payload["imbalanced"],
        "flagged": payload["flagged"],
        "observed": payload["observed"],
        "p_values": payload["p_values"],
    }, indent=2))


if __name__ == "__main__":
    main()
