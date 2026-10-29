"""Stratify the scored 506 patients. Do not score the remaining pneumonia pool.

EVALUATION_PROTOCOL.md section 2.23. Stored log-odds and manifests only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.nih_image_level_census import image_level_pools  # noqa: E402
from experiments.nih_pneumonia import (  # noqa: E402
    EXPECTED_ROWS,
    label_tokens,
    patient_order,
    rows_from_csv,
)
from experiments.validate_nih_view_shortcut import (  # noqa: E402
    _auc_bounds,
    _draws,
    _rate_bounds,
    above_chance,
    rate_difference,
    shortcut_supported,
    view_score,
)
from fpavlm.evaluation import roc_auc  # noqa: E402

SCORED = ROOT / "data" / "nih_chestxray14_sealed_confirmation" / "metadata.json"
WIDTH = ROOT / "experiments" / "results" / "nih_sealed_confirmation_auc_internvl2-2b.json"
CSV = ROOT / "data" / "nih_chestxray14" / "Data_Entry_2017_v2020.csv"
OLD = (
    ROOT / "data" / "nih_chestxray14" / "metadata.json",
    ROOT / "data" / "nih_chestxray14_replication" / "metadata.json",
    ROOT / "data" / "nih_chestxray14_confirmation" / "metadata.json",
)
HOLDOUT = ROOT / "data" / "nih_chestxray14_final_holdout" / "positives.json"
PA_RANKS = (4, 8)


def positive_stratum(finding_labels):
    """Bare means the only token is Pneumonia. Any other token is comorbid."""
    tokens = label_tokens(finding_labels)
    if "Pneumonia" not in tokens:
        raise ValueError("a positive row must contain Pneumonia")
    if tokens == ("Pneumonia",):
        return "bare"
    return "comorbid"


def comorbidity_carries(comorbid_above, bare_above):
    """The locked call needs the comorbid AUC above chance and the bare AUC not."""
    return bool(comorbid_above and not bare_above)


def _ids(path):
    return [str(row["patient_id"]) for row in json.loads(path.read_text(encoding="utf-8"))["records"]]


def _scores():
    payload = json.loads(WIDTH.read_text(encoding="utf-8"))
    if payload.get("protocol_section") != "2.22":
        raise SystemExit("section 2.22 scores are not on disk")
    by_rank = {row["rank"]: {item["patient_id"]: item for item in row["reports"]} for row in payload["ranks"]}
    if tuple(sorted(by_rank)) != PA_RANKS:
        raise SystemExit("section 2.22 does not have ranks 4 and 8")
    return payload, by_rank


def _auc(rows, scores):
    labels = [row["label_pneumonia"] for row in rows]
    patients = [row["patient_id"] for row in rows]
    if len(set(labels)) < 2:
        raise ValueError("both classes are required")
    y, groups, kept, _label_of = _draws(labels, patients)
    interval = _auc_bounds(y, groups, scores, kept)
    interval["above_chance"] = above_chance(interval["one_sided_95_lower"])
    interval["patients"] = len(rows)
    interval["pneumonia_patients"] = int(sum(labels))
    return interval


def main():
    manifest = json.loads(SCORED.read_text(encoding="utf-8"))
    records = manifest["records"]
    if len(records) != 506 or sum(row["label_pneumonia"] == 1 for row in records) != 253:
        raise SystemExit("the scored manifest is not the 253 + 253 set")
    payload, by_rank = _scores()
    for row in records:
        for rank in PA_RANKS:
            report = by_rank[rank][row["patient_id"]]
            if int(report["label_pneumonia"]) != int(row["label_pneumonia"]):
                raise SystemExit(f"labels disagree for patient {row['patient_id']}")
        left = float(by_rank[4][row["patient_id"]]["native_log_odds"])
        right = float(by_rank[8][row["patient_id"]]["native_log_odds"])
        if abs(left - right) > 1e-4:
            raise SystemExit("rank 8 native log-odds disagree with rank 4")
    native = [float(by_rank[4][row["patient_id"]]["native_log_odds"]) for row in records]
    labels = [row["label_pneumonia"] for row in records]
    if abs(roc_auc(labels, native) - payload["ranks"][0]["auc"]["auc_native"]) > 1e-6:
        raise SystemExit("native AUC disagrees with section 2.22")
    patients = [row["patient_id"] for row in records]
    y, groups, kept, label_of = _draws(labels, patients)
    ap_of = {row["patient_id"]: view_score(row["view_position"]) for row in records}
    view_auc = _auc_bounds(y, groups, [ap_of[patient] for patient in patients], kept)
    rate_lower, rate_upper = _rate_bounds(kept, label_of, ap_of)
    table = {"pneumonia_pa": 0, "pneumonia_ap": 0, "no_finding_pa": 0, "no_finding_ap": 0}
    for row in records:
        key = ("pneumonia" if row["label_pneumonia"] == 1 else "no_finding") + "_" + row["view_position"].lower()
        table[key] += 1
    difference = rate_difference(
        table["pneumonia_ap"], table["pneumonia_ap"] + table["pneumonia_pa"],
        table["no_finding_ap"], table["no_finding_ap"] + table["no_finding_pa"],
    )
    higher = rate_lower > 0
    predicts = above_chance(view_auc["one_sided_95_lower"])
    view = {
        **table,
        "ap_rate_difference": difference,
        "ap_rate_ci95_lower": rate_lower,
        "ap_rate_ci95_upper": rate_upper,
        "ap_rate_higher": higher,
        "view_auc": view_auc["auc"],
        "view_one_sided_95_lower": view_auc["one_sided_95_lower"],
        "view_predicts_pneumonia": predicts,
        "view_shortcut_supported": shortcut_supported(higher, predicts),
    }
    pa_rows = [row for row in records if row["view_position"] == "PA"]
    pa_native = _auc(pa_rows, [float(by_rank[4][row["patient_id"]]["native_log_odds"]) for row in pa_rows])
    pa_ranks = []
    for rank in PA_RANKS:
        interval = _auc(pa_rows, [float(by_rank[rank][row["patient_id"]]["oracle_log_odds"]) for row in pa_rows])
        pa_ranks.append({"rank": rank, **interval})
    negatives = [row for row in records if row["label_pneumonia"] == 0]
    strata = {}
    for name in ("bare", "comorbid"):
        positives = [
            row for row in records
            if row["label_pneumonia"] == 1 and positive_stratum(row["finding_labels"]) == name
        ]
        subset = positives + negatives
        interval = _auc(subset, [float(by_rank[4][row["patient_id"]]["native_log_odds"]) for row in subset])
        strata[name] = {"positive_patients": len(positives), **interval}
    carries = comorbidity_carries(strata["comorbid"]["above_chance"], strata["bare"]["above_chance"])
    inspected = []
    for path in OLD:
        inspected.extend(_ids(path))
    inspected.extend(row["patient_id"] for row in records)
    if len(inspected) != len(set(inspected)):
        raise SystemExit("an inspected patient is listed twice")
    csv_rows = rows_from_csv(CSV.read_text(encoding="utf-8"))
    if len(csv_rows) != EXPECTED_ROWS:
        raise SystemExit("the local NIH table is not the expected 112120 rows")
    remaining = image_level_pools(csv_rows, inspected)[1]
    remaining.sort(key=lambda row: patient_order(row["patient_id"]))
    id_digest = hashlib.sha256("\n".join(row["patient_id"] for row in remaining).encode("utf-8")).hexdigest()
    HOLDOUT.parent.mkdir(parents=True, exist_ok=True)
    holdout = {
        "schema": "nih_final_holdout_positives_v1",
        "protocol_section": "2.23",
        "role": "final_confirmation_positives",
        "images_downloaded": False,
        "auc_computed": False,
        "negative_sample_drawn": False,
        "future_negatives_per_positive": 2,
        "patient_count": len(remaining),
        "patient_id_sha256": id_digest,
        "records": [
            {"patient_id": row["patient_id"], "image_index": row["image_index"], "label_pneumonia": 1}
            for row in remaining
        ],
    }
    HOLDOUT.write_text(json.dumps(holdout, indent=2) + "\n", encoding="utf-8")
    result = {
        "schema": "nih_sealed_strata_v1",
        "protocol_section": "2.23",
        "model_loaded": False,
        "scored_patients_role": "development_analysis",
        "inspected_patients": len(inspected),
        "view": view,
        "pa_only": {"native": pa_native, "ranks": pa_ranks},
        "native_by_positive_stratum": strata,
        "comorbidity_carries_native_auc": carries,
        "task_wording": "pneumonia_label_versus_no_finding" if carries else "unchanged",
        "holdout_pneumonia_patients": len(remaining),
        "holdout_patient_id_sha256": id_digest,
        "holdout_images_downloaded": False,
        "holdout_auc_computed": False,
        "negative_sample_drawn": False,
        "noninferiority_claim": False,
        "cell_selected": False,
        "finite_state_symbol_tested": False,
        "predictor_retarget_started": False,
        "mlp_started": False,
        "rbf_grid_extended": False,
        "medgemma_ran": False,
        "prefix_grid_scored": False,
        "reconstruction_attack_started": False,
    }
    destination = ROOT / "experiments" / "results" / "nih_sealed_strata.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "view_shortcut_supported": view["view_shortcut_supported"],
        "pa_only_native_above_chance": pa_native["above_chance"],
        "comorbidity_carries_native_auc": carries,
        "task_wording": result["task_wording"],
        "holdout_pneumonia_patients": len(remaining),
        "holdout_images_downloaded": False,
        "prefix_grid_scored": False,
    }, indent=2))


if __name__ == "__main__":
    main()
