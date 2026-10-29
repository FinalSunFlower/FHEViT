"""Score oracle PCA ranks 8 and 32 on the confirmation AUC.

EVALUATION_PROTOCOL.md section 2.17. Ranks 4, 16, and 64 stay the stored
section 2.16 scores. This section does not retarget the predictor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.diagnose_release_sufficiency import _fingerprint, summarize  # noqa: E402
from experiments.fit_nih_oracle_bridge import (  # noqa: E402
    CONFIRMATION,
    DEVELOPMENT,
    _native,
    _records,
)
from experiments.nih_pneumonia import (  # noqa: E402
    CONFIRMATION_NEGATIVE,
    CONFIRMATION_POSITIVE,
    PROMPT,
    REPLICATION_NEGATIVE,
    REPLICATION_POSITIVE,
)
from experiments.validate_nih_oracle_auc import (  # noqa: E402
    NATIVE,
    WIDTH,
    _log_odds,
)
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402
from experiments.validate_pneumonia_log_odds import NEGATIVE, POSITIVE, single_token_id  # noqa: E402
from fpavlm import DiagnosticPrefixProjector, PrefixSpec  # noqa: E402
from fpavlm.evaluation import paired_auc_noninferiority  # noqa: E402

BACKEND = "internvl2-2b"
NEIGHBORS = (8, 32)
CURVE = (4, 8, 16, 32, 64)
STORED = ROOT / "experiments" / "results" / "nih_oracle_auc_internvl2-2b.json"


def curve_peak(points):
    """Highest oracle AUC on the locked five-rank curve. A tie keeps the smaller rank."""
    ranks = [rank for rank, _auc in points]
    if tuple(ranks) != CURVE or len(set(ranks)) != len(CURVE):
        raise ValueError("the locked curve is ranks 4, 8, 16, 32, and 64")
    best = max(auc for _rank, auc in points)
    return min(rank for rank, auc in points if auc == best)


def _stored_curve():
    stored = json.loads(STORED.read_text(encoding="utf-8"))
    if stored.get("protocol_section") != "2.16" or stored.get("backend") != BACKEND:
        raise SystemExit("section 2.16 oracle AUC file is not the locked InternVL2 result")
    rows = {row["rank"]: row for row in stored["ranks"]}
    if tuple(rows) != (4, 16, 64):
        raise SystemExit("section 2.16 does not have ranks 4, 16, and 64")
    return stored, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=[BACKEND], default=BACKEND)
    args = parser.parse_args()
    if args.backend != BACKEND:
        raise SystemExit("section 2.17 fits internvl2-2b only; Qwen stays the secondary line")
    torch.set_num_threads(4)
    torch.manual_seed(812)
    started = time.perf_counter()
    prior, stored_rows = _stored_curve()
    published = json.loads(NATIVE.read_text(encoding="utf-8"))
    width = json.loads(WIDTH.read_text(encoding="utf-8"))
    if width.get("backend") != BACKEND:
        raise SystemExit("section 2.7 width file is not InternVL2")
    development = _records(DEVELOPMENT, REPLICATION_POSITIVE + REPLICATION_NEGATIVE)
    confirmation = _records(CONFIRMATION, CONFIRMATION_POSITIVE + CONFIRMATION_NEGATIVE)
    dev_ids = [row["patient_id"] for row in development]
    conf_ids = [row["patient_id"] for row in confirmation]
    labels = [row["label_pneumonia"] for row in confirmation]
    if set(dev_ids) & set(conf_ids):
        raise ValueError("development and confirmation patients overlap")
    llm = load_backend(BACKEND)
    native = torch.cat([
        _native(llm, DEVELOPMENT, development, BACKEND),
        _native(llm, CONFIRMATION, confirmation, BACKEND),
    ]).float()
    train_n = len(development)
    test = native[train_n:]
    train_mean = native[:train_n].mean(0, keepdim=True)
    centered = native.reshape(len(native), -1).double() - native[:train_n].reshape(train_n, -1).double().mean(0)
    basis = torch.linalg.svd(centered[:train_n], full_matrices=False).Vh.T
    digest = hashlib.sha256("\n".join(dev_ids).encode("utf-8")).hexdigest()
    tokens, hidden = int(native.shape[-2]), int(native.shape[-1])
    positive_id = single_token_id(llm.tokenizer, POSITIVE)
    negative_id = single_token_id(llm.tokenizer, NEGATIVE)
    native_scores = [
        _log_odds(llm, test[offset].unsqueeze(0), positive_id, negative_id)
        for offset in range(len(confirmation))
    ]
    rows = []
    for rank in NEIGHBORS:
        code = (centered @ basis[:, :rank]).reshape(len(native), rank, 1).float()
        spec = PrefixSpec(
            BACKEND, (rank, 1), tokens, hidden, digest,
            _fingerprint({"release": "nih_pca_oracle_auc_neighbor", "rank": rank, "section": "2.17"}),
        )
        fitted = DiagnosticPrefixProjector(spec).fit(code[:train_n], native[:train_n], sigma=0.)
        prediction = fitted(code[train_n:])
        width_row = summarize(f"pca_rank_{rank}", prediction, test, train_mean, conf_ids, bootstrap_samples=2000)
        release_scores = [
            _log_odds(llm, prediction[offset].unsqueeze(0), positive_id, negative_id)
            for offset in range(len(confirmation))
        ]
        comparison = paired_auc_noninferiority(labels, native_scores, release_scores, conf_ids)
        if abs(comparison["auc_native"] - published["auc"]["auc_native"]) > 1e-4:
            raise SystemExit("native AUC disagrees with section 2.7")
        rows.append({
            "rank": rank,
            "mse_improvement_over_mean": width_row["mse_improvement_over_mean"],
            "auc": comparison,
            "noninferior": comparison["noninferior"],
            "reports": [
                {
                    "patient_id": conf_ids[offset],
                    "label_pneumonia": labels[offset],
                    "native_log_odds": native_scores[offset],
                    "oracle_log_odds": release_scores[offset],
                }
                for offset in range(len(confirmation))
            ],
        })
        print(json.dumps({
            "rank": rank,
            "auc_native": comparison["auc_native"],
            "auc_oracle": comparison["auc_release"],
            "one_sided_95_lower": comparison["one_sided_95_lower"],
            "noninferior": comparison["noninferior"],
        }), flush=True)
    points = []
    for rank in CURVE:
        if rank in NEIGHBORS:
            row = next(item for item in rows if item["rank"] == rank)
            points.append((rank, row["auc"]["auc_release"]))
        else:
            points.append((rank, stored_rows[rank]["auc"]["auc_release"]))
    peak = curve_peak(points)
    result = {
        "schema": "nih_oracle_auc_neighbors_v1",
        "protocol_section": "2.17",
        "backend": BACKEND,
        "qwen_fitted": False,
        "qwen_secondary_line": True,
        "development_patients": train_n,
        "confirmation_patients": len(confirmation),
        "same_confirmation_patients": True,
        "fit_saw_confirmation": False,
        "prompt": PROMPT,
        "neighbors": rows,
        "stored_section_2_16_auc": {
            str(rank): stored_rows[rank]["auc"]["auc_release"] for rank in (4, 16, 64)
        },
        "curve_peak_rank": peak,
        "rank_16_is_curve_peak": peak == 16,
        "predictor_retarget_started": False,
        "capacity_diagnostic_started": False,
        "privacy_channel_redesign_started": False,
        "mlp_started": False,
        "clinical_label_training_started": False,
        "official_test_list_read": False,
        "prior_coordinate_path_remains_open": prior["coordinate_path_remains_open"],
        "limitations": [
            "Ranks 8 and 32 are new oracle bridges. Ranks 4, 16, and 64 are the stored section 2.16 scores.",
            "The curve peak is a point-estimate comparison. It does not retarget the predictor.",
            "Token MSE is recorded and is not the acceptance rule.",
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    destination = ROOT / "experiments" / "results" / "nih_oracle_auc_neighbors_internvl2-2b.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "curve_peak_rank": peak,
        "rank_16_is_curve_peak": peak == 16,
        "predictor_retarget_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
