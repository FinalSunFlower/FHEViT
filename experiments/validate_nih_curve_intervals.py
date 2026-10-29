"""Patient-bootstrap bands for the stored oracle AUC curve.

EVALUATION_PROTOCOL.md section 2.18. Stored log-odds only. No model, no
refit, and no rank is selected. The confirmation patients stay a selection set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpavlm.evaluation import roc_auc  # noqa: E402

RANKS = (4, 8, 16, 32, 64)
DRAWS = 2000
SEED = 42
STORED = {
    4: ROOT / "experiments" / "results" / "nih_oracle_auc_internvl2-2b.json",
    16: ROOT / "experiments" / "results" / "nih_oracle_auc_internvl2-2b.json",
    64: ROOT / "experiments" / "results" / "nih_oracle_auc_internvl2-2b.json",
    8: ROOT / "experiments" / "results" / "nih_oracle_auc_neighbors_internvl2-2b.json",
    32: ROOT / "experiments" / "results" / "nih_oracle_auc_neighbors_internvl2-2b.json",
}
SECTION = {4: "2.16", 8: "2.17", 16: "2.16", 32: "2.17", 64: "2.16"}


def rank_pairs(ranks=RANKS):
    """Smaller rank first. The locked curve is the only accepted order."""
    if tuple(ranks) != RANKS:
        raise ValueError("the locked curve is ranks 4, 8, 16, 32, and 64")
    return [(ranks[i], ranks[j]) for i in range(len(ranks)) for j in range(i + 1, len(ranks))]


def includes_zero(lower, upper):
    """A two-sided interval includes zero when it touches or straddles zero."""
    if upper < lower:
        raise ValueError("an interval upper bound is below its lower bound")
    return lower <= 0 <= upper


def _reports(payload, rank):
    rows = payload["ranks"] if "ranks" in payload else payload["neighbors"]
    found = [row for row in rows if row["rank"] == rank]
    if len(found) != 1:
        raise SystemExit(f"rank {rank} is not stored once")
    return found[0]


def _aligned(payloads):
    anchor = _reports(payloads[4], 4)["reports"]
    patients = [row["patient_id"] for row in anchor]
    labels = [int(row["label_pneumonia"]) for row in anchor]
    native = [float(row["native_log_odds"]) for row in anchor]
    oracle = {}
    for rank in RANKS:
        reports = _reports(payloads[rank], rank)["reports"]
        by_id = {row["patient_id"]: row for row in reports}
        if set(by_id) != set(patients) or len(by_id) != len(patients):
            raise SystemExit(f"rank {rank} patients disagree with rank 4")
        if [int(by_id[patient]["label_pneumonia"]) for patient in patients] != labels:
            raise SystemExit(f"rank {rank} labels disagree with rank 4")
        other_native = [float(by_id[patient]["native_log_odds"]) for patient in patients]
        if any(abs(left - right) > 1e-4 for left, right in zip(native, other_native)):
            raise SystemExit(f"rank {rank} native log-odds disagree with rank 4")
        oracle[rank] = [float(by_id[patient]["oracle_log_odds"]) for patient in patients]
        stored = _reports(payloads[rank], rank)["auc"]["auc_release"]
        if abs(roc_auc(labels, oracle[rank]) - stored) > 1e-6:
            raise SystemExit(f"rank {rank} oracle AUC disagrees with the stored point")
    return patients, np.asarray(labels), np.asarray(native, dtype=float), {
        rank: np.asarray(scores, dtype=float) for rank, scores in oracle.items()
    }


def _interval(samples):
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def curve_intervals(labels, native, oracle, patients, *, draws=DRAWS, seed=SEED):
    """Shared patient bootstrap. Pairwise differences use the same draws."""
    y = np.asarray(labels)
    groups = np.asarray(patients)
    ids = np.unique(groups)
    if len(ids) < 2 or draws < 100:
        raise ValueError("at least two patients and 100 draws are required")
    index_groups = {patient: np.flatnonzero(groups == patient) for patient in ids}
    rng = np.random.default_rng(seed)
    kept = {rank: [] for rank in oracle}
    gaps = {rank: [] for rank in oracle}
    for _ in range(draws):
        chosen = rng.choice(ids, size=len(ids), replace=True)
        rows = np.concatenate([index_groups[patient] for patient in chosen])
        if len(np.unique(y[rows])) < 2:
            continue
        native_auc = roc_auc(y[rows], np.asarray(native)[rows])
        for rank, scores in oracle.items():
            value = roc_auc(y[rows], np.asarray(scores)[rows])
            kept[rank].append(value)
            gaps[rank].append(value - native_auc)
    if any(len(samples) < 100 for samples in kept.values()):
        raise ValueError("fewer than 100 bootstrap draws contained both classes")
    bands = {}
    for rank in oracle:
        samples = np.asarray(kept[rank], dtype=float)
        gap = np.asarray(gaps[rank], dtype=float)
        lower, upper = _interval(samples)
        bands[rank] = {
            "auc": float(roc_auc(y, oracle[rank])),
            "ci95_lower": lower,
            "ci95_upper": upper,
            "difference_bootstrap_sd": float(gap.std(ddof=1)),
            "valid_draws": int(len(samples)),
        }
    pairs = []
    for left, right in rank_pairs():
        delta = np.asarray(kept[left], dtype=float) - np.asarray(kept[right], dtype=float)
        lower, upper = _interval(delta)
        pairs.append({
            "smaller_rank": left,
            "larger_rank": right,
            "difference": bands[left]["auc"] - bands[right]["auc"],
            "ci95_lower": lower,
            "ci95_upper": upper,
            "includes_zero": includes_zero(lower, upper),
        })
    return bands, pairs


def main():
    payloads = {}
    for rank, path in STORED.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_section") != SECTION[rank] or payload.get("backend") != "internvl2-2b":
            raise SystemExit(f"rank {rank} is not the locked InternVL2 file")
        payloads[rank] = payload
    patients, labels, native, oracle = _aligned(payloads)
    bands, pairs = curve_intervals(labels, native, oracle, patients)
    result = {
        "schema": "nih_oracle_auc_curve_intervals_v1",
        "protocol_section": "2.18",
        "backend": "internvl2-2b",
        "confirmation_is_selection_set": True,
        "noninferiority_claim": False,
        "model_loaded": False,
        "bridge_refit": False,
        "new_confirmation_sampled": False,
        "patients": len(patients),
        "bootstrap_samples": DRAWS,
        "seed": SEED,
        "ranks": [{**{"rank": rank}, **bands[rank]} for rank in RANKS],
        "pairs": pairs,
        "pairs_including_zero": sum(row["includes_zero"] for row in pairs),
        "pair_count": len(pairs),
        "rank_selected": False,
        "predictor_retarget_started": False,
        "mlp_started": False,
        "privacy_channel_redesign_started": False,
        "regional_release_started": False,
        "limitations": [
            "These patients already compared the five ranks, so the intervals are not a fresh non-inferiority claim.",
            "Including zero is a description of the interval. It does not select a rank.",
        ],
    }
    destination = ROOT / "experiments" / "results" / "nih_oracle_auc_curve_intervals.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "pairs_including_zero": result["pairs_including_zero"],
        "pair_count": result["pair_count"],
        "difference_bootstrap_sd": {str(rank): bands[rank]["difference_bootstrap_sd"] for rank in RANKS},
        "rank_selected": False,
    }, indent=2))


if __name__ == "__main__":
    main()
