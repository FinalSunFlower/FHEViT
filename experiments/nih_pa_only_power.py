"""Count PA-only patients still in reserve, and the power of that count.

EVALUATION_PROTOCOL.md section 2.25. No image is downloaded and no
confirmation patient is scored.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.nih_image_level_census import image_level_pools  # noqa: E402
from experiments.nih_pneumonia import EXPECTED_ROWS, rows_from_csv  # noqa: E402
from experiments.nih_sample_size import power_noninferior, se_at_n  # noqa: E402
from fpavlm.evaluation import roc_auc  # noqa: E402

DRAWS = 2000
SEED = 42
DELTAS = (0.0, 0.03)
SCORED = ROOT / "data" / "nih_chestxray14_sealed_confirmation" / "metadata.json"
WIDTH = ROOT / "experiments" / "results" / "nih_sealed_confirmation_auc_internvl2-2b.json"
STRATA = ROOT / "experiments" / "results" / "nih_sealed_strata.json"
HOLDOUT = ROOT / "data" / "nih_chestxray14_final_holdout" / "positives.json"
CSV = ROOT / "data" / "nih_chestxray14" / "Data_Entry_2017_v2020.csv"
OLD = (
    ROOT / "data" / "nih_chestxray14" / "metadata.json",
    ROOT / "data" / "nih_chestxray14_replication" / "metadata.json",
    ROOT / "data" / "nih_chestxray14_confirmation" / "metadata.json",
)
_NORMAL = statistics.NormalDist()


def include_in_pa_pool(chosen_view):
    """The chosen film stands. Only PA enters the pool. A later PA is not promoted."""
    if chosen_view not in {"PA", "AP"}:
        raise ValueError("chosen view must be PA or AP")
    return chosen_view == "PA"


def planning_se(rank4, rank8):
    """The larger of the two locked PA-only standard deviations."""
    if rank4 <= 0 or rank8 <= 0 or not math.isfinite(rank4) or not math.isfinite(rank8):
        raise ValueError("two positive standard deviations are required")
    return float(max(rank4, rank8))


def required_positives(se, n_pos_ref, delta, *, margin=0.05, power=0.80):
    """Smallest PA-positive count at the target power. Negatives are not drawn."""
    if se <= 0 or n_pos_ref < 2 or delta + margin <= 0 or not 0 < power < 1:
        raise ValueError("the planning inputs do not give a finite positive count")
    se_needed = (delta + margin) / (_NORMAL.inv_cdf(0.95) + _NORMAL.inv_cdf(power))
    return int(math.ceil(n_pos_ref * (se / se_needed) ** 2))


def _difference_sd(labels, native, release, patients):
    y = np.asarray(labels)
    left = np.asarray(native, dtype=float)
    right = np.asarray(release, dtype=float)
    groups = np.asarray(patients)
    ids = np.unique(groups)
    index = {patient: np.flatnonzero(groups == patient) for patient in ids}
    rng = np.random.default_rng(SEED)
    gaps = []
    for _ in range(DRAWS):
        chosen = rng.choice(ids, size=len(ids), replace=True)
        rows = np.concatenate([index[patient] for patient in chosen])
        if len(np.unique(y[rows])) < 2:
            continue
        gaps.append(roc_auc(y[rows], right[rows]) - roc_auc(y[rows], left[rows]))
    if len(gaps) < 100:
        raise ValueError("fewer than 100 bootstrap draws contained both classes")
    return float(np.asarray(gaps, dtype=float).std(ddof=1))


def _views_by_image(rows):
    found = {}
    for row in rows:
        image = row["image_index"]
        view = (row["view_position"] or "").strip()
        if image in found and found[image] != view:
            raise SystemExit(f"{image} has two view values")
        found[image] = view
    return found


def main():
    strata = json.loads(STRATA.read_text(encoding="utf-8"))
    stored_pa = strata["pa_only"]["native"]["auc"]
    manifest = json.loads(SCORED.read_text(encoding="utf-8"))
    width = json.loads(WIDTH.read_text(encoding="utf-8"))
    by_rank = {row["rank"]: {item["patient_id"]: item for item in row["reports"]} for row in width["ranks"]}
    pa_rows = [row for row in manifest["records"] if include_in_pa_pool(row["view_position"])]
    labels = [row["label_pneumonia"] for row in pa_rows]
    patients = [row["patient_id"] for row in pa_rows]
    native = [float(by_rank[4][row["patient_id"]]["native_log_odds"]) for row in pa_rows]
    if abs(roc_auc(labels, native) - stored_pa) > 1e-6:
        raise SystemExit("PA-only native AUC disagrees with section 2.23")
    n_pos_ref = int(sum(labels))
    deviations = {
        rank: _difference_sd(
            labels, native,
            [float(by_rank[rank][row["patient_id"]]["oracle_log_odds"]) for row in pa_rows],
            patients,
        )
        for rank in (4, 8)
    }
    se = planning_se(deviations[4], deviations[8])
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        "\n".join(row["patient_id"] for row in holdout["records"]).encode("utf-8")
    ).hexdigest()
    if digest != holdout["patient_id_sha256"] or holdout["patient_count"] != 598:
        raise SystemExit("the reserved pneumonia list is not the section 2.23 file")
    csv_rows = rows_from_csv(CSV.read_text(encoding="utf-8"))
    if len(csv_rows) != EXPECTED_ROWS:
        raise SystemExit("the local NIH table is not the expected 112120 rows")
    views = _views_by_image(csv_rows)
    reserved_pa = 0
    reserved_ap = 0
    for row in holdout["records"]:
        view = views.get(row["image_index"])
        if view not in {"PA", "AP"}:
            raise SystemExit(f"{row['image_index']} has no PA or AP view")
        if include_in_pa_pool(view):
            reserved_pa += 1
        else:
            reserved_ap += 1
    inspected = []
    for path in (*OLD, SCORED):
        inspected.extend(str(row["patient_id"]) for row in json.loads(path.read_text(encoding="utf-8"))["records"])
    if len(inspected) != len(set(inspected)):
        raise SystemExit("an inspected patient is listed twice")
    unused_negatives = image_level_pools(csv_rows, inspected)[0]
    negative_pa = sum(include_in_pa_pool(row["view_position"]) for row in unused_negatives)
    plans = []
    for delta in DELTAS:
        needed = required_positives(se, n_pos_ref, delta)
        plans.append({
            "true_difference": delta,
            "required_pa_positives": needed,
            "power_at_reserved_pa": power_noninferior(se_at_n(se, reserved_pa, n_pos_ref), delta),
        })
    allows = plans[0]["power_at_reserved_pa"] >= 0.80
    result = {
        "schema": "nih_pa_only_power_v1",
        "protocol_section": "2.25",
        "development_pa_patients": len(pa_rows),
        "development_pa_positives": n_pos_ref,
        "development_pa_native_auc": float(roc_auc(labels, native)),
        "difference_bootstrap_sd": deviations,
        "planning_se": se,
        "reserved_pneumonia_pa": reserved_pa,
        "reserved_pneumonia_ap": reserved_ap,
        "unused_no_finding_pa": negative_pa,
        "plans": plans,
        "power_allows_sampling": allows,
        "confirmation_sample_drawn": False,
        "images_downloaded": False,
        "holdout_auc_computed": False,
        "grid_scored": False,
        "limitations": [
            "The standard error scales with the PA-positive count. The 1:2 negative ratio is not sampled.",
            "An AP chosen image is counted and is not replaced by a later PA image.",
        ],
    }
    destination = ROOT / "experiments" / "results" / "nih_pa_only_power.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "planning_se": se,
        "reserved_pneumonia_pa": reserved_pa,
        "reserved_pneumonia_ap": reserved_ap,
        "unused_no_finding_pa": negative_pa,
        "plans": plans,
        "power_allows_sampling": allows,
        "confirmation_sample_drawn": False,
        "grid_scored": False,
    }, indent=2))


if __name__ == "__main__":
    main()
