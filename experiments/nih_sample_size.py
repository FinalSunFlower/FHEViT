"""Sample size for the oracle AUC margin, and the remaining NIH pool.

EVALUATION_PROTOCOL.md section 2.19. The census does not write a manifest
and does not download images.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.nih_pneumonia import (  # noqa: E402
    EXPECTED_ROWS,
    classify_patient,
    label_tokens,
    rows_from_csv,
)

CURVE = ROOT / "experiments" / "results" / "nih_oracle_auc_curve_intervals.json"
CSV = ROOT / "data" / "nih_chestxray14" / "Data_Entry_2017_v2020.csv"
MANIFESTS = (
    ROOT / "data" / "nih_chestxray14" / "metadata.json",
    ROOT / "data" / "nih_chestxray14_replication" / "metadata.json",
    ROOT / "data" / "nih_chestxray14_confirmation" / "metadata.json",
)
MARGIN = 0.05
N_REF = 72
DELTAS = (0.0, 0.03)
_NORMAL = statistics.NormalDist()
Z95 = _NORMAL.inv_cdf(0.95)
Z80 = _NORMAL.inv_cdf(0.80)


def planning_se(values):
    """Median of the five locked bootstrap standard deviations."""
    if len(values) != 5 or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("five positive standard errors are required")
    return float(statistics.median(values))


def power_noninferior(se, delta, margin=MARGIN):
    """Normal power of a one-sided 95% lower bound above -margin."""
    if se <= 0 or margin <= 0 or not math.isfinite(delta):
        raise ValueError("a positive standard error and margin are required")
    return float(_NORMAL.cdf(-Z95 + (delta + margin) / se))


def planned_even_n(se, delta, *, n_ref=N_REF, margin=MARGIN, power=0.80):
    """Smallest even n with the locked normal approximation at the target power."""
    if se <= 0 or n_ref < 2 or not 0 < power < 1 or delta + margin <= 0:
        raise ValueError("the planning inputs do not give a finite sample size")
    z_power = _NORMAL.inv_cdf(power)
    se_needed = (delta + margin) / (Z95 + z_power)
    raw = n_ref * (se / se_needed) ** 2
    return int(math.ceil(raw / 2.0)) * 2


def se_at_n(se_ref, n, n_ref=N_REF):
    if se_ref <= 0 or n < 2 or n_ref < 2:
        raise ValueError("sample size scaling needs positive inputs")
    return se_ref * math.sqrt(n_ref / n)


def eligible_counts(rows, exclude_ids):
    """Full eligible pools. Nothing is truncated to a probe size."""
    excluded = {str(patient) for patient in exclude_ids}
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["patient_id"]).strip(), []).append(row)
    counts = {0: 0, 1: 0}
    for patient, records in grouped.items():
        if patient in excluded:
            continue
        decision = classify_patient(records)
        if decision is not None:
            counts[decision[0]] += 1
    return {"pneumonia": counts[1], "no_finding": counts[0]}


def pneumonia_token_counts(rows):
    images = 0
    patients = set()
    for row in rows:
        if "Pneumonia" in label_tokens(row["finding_labels"]):
            images += 1
            patients.add(str(row["patient_id"]).strip())
    return {"pneumonia_image_rows": images, "patients_with_a_pneumonia_token": len(patients)}


def _excluded():
    ids = []
    for path in MANIFESTS:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        ids.extend(str(row["patient_id"]) for row in manifest["records"])
    if len(ids) != len(set(ids)):
        raise SystemExit("a patient appears in more than one sealed manifest")
    return ids


def main():
    curve = json.loads(CURVE.read_text(encoding="utf-8"))
    if curve.get("protocol_section") != "2.18":
        raise SystemExit("section 2.18 intervals are not on disk")
    standard_errors = [row["difference_bootstrap_sd"] for row in curve["ranks"]]
    if [row["rank"] for row in curve["ranks"]] != [4, 8, 16, 32, 64]:
        raise SystemExit("section 2.18 does not have the locked five ranks")
    se = planning_se(standard_errors)
    plans = []
    for delta in DELTAS:
        size = planned_even_n(se, delta)
        plans.append({
            "true_difference": delta,
            "planned_even_n": size,
            "per_class": size // 2,
            "power_at_72": power_noninferior(se, delta),
            "power_at_plan": power_noninferior(se_at_n(se, size), delta),
        })
    rows = rows_from_csv(CSV.read_text(encoding="utf-8"))
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit("the local NIH table is not the expected 112120 rows")
    excluded = _excluded()
    remaining = eligible_counts(rows, excluded)
    tokens = pneumonia_token_counts(rows)
    result = {
        "schema": "nih_oracle_auc_power_v1",
        "protocol_section": "2.19",
        "margin": MARGIN,
        "z_0_95": Z95,
        "z_0_80": Z80,
        "reference_patients": N_REF,
        "standard_errors": standard_errors,
        "planning_se": se,
        "plans": plans,
        "table_rows": len(rows),
        "excluded_patients": len(excluded),
        "pneumonia_token_counts": tokens,
        "remaining_eligible": remaining,
        "new_confirmation_manifest_written": False,
        "images_downloaded": False,
        "predictor_retarget_started": False,
        "limitations": [
            "The normal approximation scales the 72-patient bootstrap standard error with the square root of n.",
            "Remaining patients are eligible under section 2.5. They are not a sampled confirmation set.",
        ],
    }
    destination = ROOT / "experiments" / "results" / "nih_oracle_auc_power.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "planning_se": se,
        "plans": plans,
        "remaining_eligible": remaining,
        "pneumonia_token_counts": tokens,
        "images_downloaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
