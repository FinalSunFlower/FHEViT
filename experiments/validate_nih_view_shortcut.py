"""Does PA versus AP separate pneumonia in the stored NIH slices?

EVALUATION_PROTOCOL.md section 2.21. Manifests and stored log-odds only.
No model is loaded. Component scores are not recreated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpavlm.evaluation import roc_auc  # noqa: E402

DRAWS = 2000
SEED = 42
PA_RANKS = (4, 8, 16)
DEVELOPMENT = ROOT / "data" / "nih_chestxray14_replication" / "metadata.json"
CONFIRMATION = ROOT / "data" / "nih_chestxray14_confirmation" / "metadata.json"
WIDTH = ROOT / "experiments" / "results" / "nih_oracle_auc_internvl2-2b.json"
NEIGHBORS = ROOT / "experiments" / "results" / "nih_oracle_auc_neighbors_internvl2-2b.json"
COORDINATES = ROOT / "experiments" / "results" / "nih_oracle_coordinates_internvl2-2b.npz"


def view_score(view):
    """AP is the positive view score. PA is zero."""
    if view == "AP":
        return 1.0
    if view == "PA":
        return 0.0
    raise ValueError("view must be PA or AP")


def rate_difference(ap_pneumonia, n_pneumonia, ap_negative, n_negative):
    if n_pneumonia < 1 or n_negative < 1:
        raise ValueError("both classes are required")
    return ap_pneumonia / n_pneumonia - ap_negative / n_negative


def shortcut_supported(rate_higher, view_predicts):
    """Both locked conditions are required. Either one alone does not qualify."""
    return bool(rate_higher and view_predicts)


def above_chance(lower):
    if not np.isfinite(lower):
        raise ValueError("the lower bound must be finite")
    return bool(lower > 0.5)


def _manifest(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["records"]
    if len(rows) != 72 or len({row["patient_id"] for row in rows}) != 72:
        raise SystemExit(f"{path.parent.name} is not the sealed 72")
    for row in rows:
        if row["view_position"] not in {"PA", "AP"} or row["label_pneumonia"] not in {0, 1}:
            raise SystemExit(f"{path.parent.name} has a row outside the locked labels")
    return rows


def _draws(labels, patients):
    """One unstratified patient bootstrap. A one-class draw is skipped."""
    y = np.asarray(labels)
    groups = np.asarray(patients)
    ids = np.unique(groups)
    if len(ids) < 2 or len(set(np.asarray(y).tolist())) < 2:
        raise ValueError("both classes and two patients are required")
    index_groups = {patient: np.flatnonzero(groups == patient) for patient in ids}
    label_of = {patient: int(y[index_groups[patient][0]]) for patient in ids}
    rng = np.random.default_rng(SEED)
    kept = []
    for _ in range(DRAWS):
        chosen = rng.choice(ids, size=len(ids), replace=True)
        rows = np.concatenate([index_groups[patient] for patient in chosen])
        if len(np.unique(y[rows])) < 2:
            continue
        kept.append(chosen)
    if len(kept) < 100:
        raise ValueError("fewer than 100 bootstrap draws contained both classes")
    return y, groups, kept, label_of


def _auc_bounds(y, groups, scores, kept):
    score = np.asarray(scores, dtype=float)
    index = {patient: int(np.flatnonzero(groups == patient)[0]) for patient in np.unique(groups)}
    samples = [roc_auc(y[[index[patient] for patient in chosen]], score[[index[patient] for patient in chosen]])
               for chosen in kept]
    samples = np.asarray(samples, dtype=float)
    point_rows = [index[patient] for patient in index]
    return {
        "auc": float(roc_auc(y[point_rows], score[point_rows])),
        "one_sided_95_lower": float(np.quantile(samples, 0.05)),
        "valid_draws": int(len(samples)),
    }


def _rate_bounds(kept, label_of, ap_of):
    gaps = []
    for chosen in kept:
        positive = [ap_of[patient] for patient in chosen if label_of[patient] == 1]
        negative = [ap_of[patient] for patient in chosen if label_of[patient] == 0]
        gaps.append(float(np.mean(positive) - np.mean(negative)))
    samples = np.asarray(gaps, dtype=float)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _crosstab(rows):
    table = {"pneumonia_pa": 0, "pneumonia_ap": 0, "no_finding_pa": 0, "no_finding_ap": 0}
    for row in rows:
        key = ("pneumonia" if row["label_pneumonia"] == 1 else "no_finding") + "_" + row["view_position"].lower()
        table[key] += 1
    difference = rate_difference(
        table["pneumonia_ap"], table["pneumonia_ap"] + table["pneumonia_pa"],
        table["no_finding_ap"], table["no_finding_ap"] + table["no_finding_pa"],
    )
    labels = [row["label_pneumonia"] for row in rows]
    scores = [view_score(row["view_position"]) for row in rows]
    patients = [row["patient_id"] for row in rows]
    y, groups, kept, label_of = _draws(labels, patients)
    ap_of = {row["patient_id"]: view_score(row["view_position"]) for row in rows}
    interval = _auc_bounds(y, groups, scores, kept)
    rate_lower, rate_upper = _rate_bounds(kept, label_of, ap_of)
    higher = rate_lower > 0
    predicts = above_chance(interval["one_sided_95_lower"])
    return {
        **table,
        "ap_rate_difference": difference,
        "ap_rate_ci95_lower": rate_lower,
        "ap_rate_ci95_upper": rate_upper,
        "ap_rate_higher": higher,
        "view_auc": interval["auc"],
        "view_one_sided_95_lower": interval["one_sided_95_lower"],
        "view_predicts_pneumonia": predicts,
        "view_shortcut_supported": shortcut_supported(higher, predicts),
        "valid_draws": interval["valid_draws"],
    }


def _stored_scores():
    width = json.loads(WIDTH.read_text(encoding="utf-8"))
    neighbors = json.loads(NEIGHBORS.read_text(encoding="utf-8"))
    by_rank = {}
    for row in width["ranks"]:
        if row["rank"] in PA_RANKS:
            by_rank[row["rank"]] = {item["patient_id"]: item for item in row["reports"]}
    for row in neighbors["neighbors"]:
        if row["rank"] in PA_RANKS:
            by_rank[row["rank"]] = {item["patient_id"]: item for item in row["reports"]}
    if tuple(sorted(by_rank)) != PA_RANKS:
        raise SystemExit("stored scores do not have ranks 4, 8, and 16")
    return by_rank


def _pa_only(confirmation, stored):
    rows = [row for row in confirmation if row["view_position"] == "PA"]
    labels = [row["label_pneumonia"] for row in rows]
    if len(set(labels)) < 2:
        raise SystemExit("the PA subset does not contain both classes")
    patients = [row["patient_id"] for row in rows]
    y, groups, kept, _label_of = _draws(labels, patients)
    native = [float(stored[4][row["patient_id"]]["native_log_odds"]) for row in rows]
    native_interval = _auc_bounds(y, groups, native, kept)
    ranks = []
    for rank in PA_RANKS:
        scores = [float(stored[rank][row["patient_id"]]["oracle_log_odds"]) for row in rows]
        anchor = stored[4]
        for row in rows:
            left = float(stored[rank][row["patient_id"]]["native_log_odds"])
            right = float(anchor[row["patient_id"]]["native_log_odds"])
            if abs(left - right) > 1e-4:
                raise SystemExit(f"rank {rank} native log-odds disagree on a PA patient")
        interval = _auc_bounds(y, groups, scores, kept)
        ranks.append({
            "rank": rank,
            "auc": interval["auc"],
            "one_sided_95_lower": interval["one_sided_95_lower"],
            "above_chance": above_chance(interval["one_sided_95_lower"]),
            "patients": len(rows),
            "pneumonia_patients": int(sum(labels)),
        })
    return {
        "patients": len(rows),
        "pneumonia_patients": int(sum(labels)),
        "no_finding_patients": int(len(labels) - sum(labels)),
        "native_auc": native_interval["auc"],
        "native_one_sided_95_lower": native_interval["one_sided_95_lower"],
        "native_above_chance": above_chance(native_interval["one_sided_95_lower"]),
        "ranks": ranks,
    }


def main():
    if COORDINATES.is_file():
        raise SystemExit("a coordinate file is present; section 2.21 does not read features")
    development = _crosstab(_manifest(DEVELOPMENT))
    confirmation = _crosstab(_manifest(CONFIRMATION))
    pa_only = _pa_only(_manifest(CONFIRMATION), _stored_scores())
    result = {
        "schema": "nih_view_shortcut_v1",
        "protocol_section": "2.21",
        "model_loaded": False,
        "component_correlations_computed": False,
        "coordinates_on_disk": False,
        "development": development,
        "confirmation": confirmation,
        "pa_only": pa_only,
        "noninferiority_claim": False,
        "rank_selected": False,
        "direction_nulling_started": False,
        "predictor_retarget_started": False,
        "mlp_started": False,
        "privacy_channel_redesign_started": False,
        "regional_release_started": False,
        "limitations": [
            "Component scores were not archived. This section does not reload the vision tower to recreate them.",
            "The confirmation patients are a selection set. These AUCs are not a non-inferiority claim.",
        ],
    }
    destination = ROOT / "experiments" / "results" / "nih_view_shortcut.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "development_shortcut": development["view_shortcut_supported"],
        "confirmation_shortcut": confirmation["view_shortcut_supported"],
        "pa_only_native_above_chance": pa_only["native_above_chance"],
        "component_correlations_computed": False,
        "direction_nulling_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
