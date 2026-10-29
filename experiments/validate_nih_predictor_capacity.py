"""Can a frozen DenseNet predict InternVL2's oracle PCA-4 on the NIH split?

EVALUATION_PROTOCOL.md section 2.8. The basis and every ridge see development
patients only. Confirmation patients are the evaluation set at every training
size. No privacy noise, no discretization, and no new finite-state alphabet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.diagnose_release_sufficiency import _fingerprint, summarize  # noqa: E402
from experiments.fit_nih_oracle_bridge import CONFIRMATION, DEVELOPMENT, _native, _records  # noqa: E402
from experiments.nih_pneumonia import (  # noqa: E402
    CONFIRMATION_NEGATIVE,
    CONFIRMATION_POSITIVE,
    NEGATIVE_LIMIT,
    POSITIVE_LIMIT,
    REPLICATION_NEGATIVE,
    REPLICATION_POSITIVE,
)
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402
from experiments.validate_pretrained_anchor import (  # noqa: E402
    ALPHAS,
    RELEASE_RANK,
    RIDGE,
    WEIGHTS,
    densenet_features,
    fit_prefix,
    plot_curve,
    spearman,
)
from fpavlm import PrefixSpec  # noqa: E402
from fpavlm.nested_ridge import predict_ridge, select_ridge  # noqa: E402

PATIENT_COUNTS = (4, 8, 12, 16, 20, 36, 72)
PROBE = ROOT / "data" / "nih_chestxray14"
BACKEND = "internvl2-2b"


def _ids(path, expected):
    records = _records(path, expected)
    ids = [row["patient_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name} repeats a patient")
    return records, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=[BACKEND], default=BACKEND)
    args = parser.parse_args()
    if args.backend != BACKEND:
        raise SystemExit("section 2.8 fits internvl2-2b only")
    torch.set_num_threads(4)
    torch.manual_seed(812)
    started = time.perf_counter()
    development, dev_ids = _ids(DEVELOPMENT, REPLICATION_POSITIVE + REPLICATION_NEGATIVE)
    confirmation, conf_ids = _ids(CONFIRMATION, CONFIRMATION_POSITIVE + CONFIRMATION_NEGATIVE)
    probe, probe_ids = _ids(PROBE, POSITIVE_LIMIT + NEGATIVE_LIMIT)
    if set(dev_ids) & set(conf_ids) or set(dev_ids) & set(probe_ids) or set(conf_ids) & set(probe_ids):
        raise ValueError("the NIH slices are not disjoint")
    del probe
    paths = [DEVELOPMENT / row["relative_path"] for row in development]
    paths += [CONFIRMATION / row["relative_path"] for row in confirmation]
    features, weight_sha = densenet_features(paths)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    llm = load_backend(BACKEND)
    native = torch.cat([
        _native(llm, DEVELOPMENT, development, BACKEND),
        _native(llm, CONFIRMATION, confirmation, BACKEND),
    ]).float()
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_n = len(development)
    if features.shape != (len(native), 1024):
        raise ValueError("DenseNet features do not align with the native rows")
    flat = native.reshape(len(native), -1).double()
    centered = flat - flat[:train_n].mean(0)
    basis = torch.linalg.svd(centered[:train_n], full_matrices=False).Vh.T[:, :RELEASE_RANK]
    oracle = (centered @ basis).float()
    tokens, hidden = int(native.shape[-2]), int(native.shape[-1])
    digest = hashlib.sha256("\n".join(dev_ids).encode("utf-8")).hexdigest()
    order = np.random.default_rng(812).permutation(sorted(dev_ids)).tolist()
    if len(order) != PATIENT_COUNTS[-1]:
        raise ValueError("the learning curve is declared for the 72 development patients")

    def spec_for(name):
        return PrefixSpec(
            BACKEND, (RELEASE_RANK, 1), tokens, hidden, digest,
            _fingerprint({"release": name, "rank": RELEASE_RANK, "section": "2.8"}),
        )

    curve = []
    dev_index = {patient: i for i, patient in enumerate(dev_ids)}
    for count in PATIENT_COUNTS:
        chosen = order[:count]
        subset = [dev_index[patient] for patient in chosen]
        print(json.dumps({"training_patients": count}), flush=True)
        token_mean = native[subset].mean(0, keepdim=True)
        coordinate_mean = oracle[subset].mean(0, keepdim=True)
        model = select_ridge(features[subset], oracle[subset].numpy(), chosen, ALPHAS)
        predicted = torch.as_tensor(predict_ridge(features, model), dtype=torch.float32)
        test = list(range(train_n, len(native)))
        coordinate = summarize(
            "densenet_coordinates", predicted[test], oracle[test], coordinate_mean, conf_ids,
            bootstrap_samples=2000)
        prefix = summarize(
            "densenet_prefix", fit_prefix(predicted, native, subset, test, spec_for(f"densenet-{count}")),
            native[test], token_mean, conf_ids, bootstrap_samples=2000)
        oracle_prefix = summarize(
            "oracle_prefix", fit_prefix(oracle, native, subset, test, spec_for(f"oracle-{count}")),
            native[test], token_mean, conf_ids, bootstrap_samples=2000)
        curve.append({
            "training_patients": count,
            "patient_ids": chosen,
            "alpha": None if model["intercept_only"] else model["alpha"],
            "intercept_only": model["intercept_only"],
            "alpha_at_boundary": model["alpha_at_boundary"],
            "coordinate": coordinate,
            "prefix": prefix,
            "oracle_prefix": oracle_prefix,
        })
    final = curve[-1]
    if final["training_patients"] != 72:
        raise RuntimeError("the gate row is not the 72-patient fit")
    gains = [row["prefix"]["mse_improvement_over_mean"] for row in curve]
    intercepts = all(row["intercept_only"] for row in curve)
    coordinate_hit = bool(final["coordinate"]["beats_mean_significantly"])
    prefix_hit = bool(final["prefix"]["beats_mean_significantly"])
    if prefix_hit:
        branch = "predicted_coordinates_beat_the_token_mean"
    elif final["intercept_only"] or not coordinate_hit:
        branch = "no_coordinate_signal"
    else:
        branch = "coordinates_fit_but_prefix_does_not"
    result = {
        "schema": "nih_predictor_capacity_v1",
        "protocol_section": "2.8",
        "backend": BACKEND,
        "encoder": WEIGHTS,
        "encoder_sha256": weight_sha,
        "encoder_trained": False,
        "feature_dim": 1024,
        "release_rank": RELEASE_RANK,
        "patient_counts": list(PATIENT_COUNTS),
        "patient_order_seed": 812,
        "penalty": "patient_leave_one_out_on_standardized_features",
        "prefix_ridge": RIDGE,
        "development_patients": len(development),
        "confirmation_patients": len(confirmation),
        "fit_saw_confirmation": False,
        "privacy_noise": False,
        "discretized": False,
        "finite_state_alphabet_started": False,
        "qwen_fitted": False,
        "medgemma_ran": False,
        "official_test_list_read": False,
        "curve": curve,
        "intercept_only_at_every_size": intercepts,
        "intercept_only_at_72": bool(final["intercept_only"]),
        "coordinate_beats_mean": coordinate_hit,
        "prefix_beats_mean": prefix_hit,
        "branch": branch,
        "spearman_prefix_vs_patients": None if intercepts else spearman(PATIENT_COUNTS, gains),
        "limitations": [
            "The PCA basis is fit once on all 72 development patients so the target stays fixed.",
            "The encoder is frozen and plaintext. This is not an encrypted release.",
            "No training size was selected after seeing the confirmation scores.",
            "A new finite-state alphabet is not started from this result.",
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    destination = ROOT / "experiments" / "results" / "nih_predictor_capacity_internvl2-2b.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_curve(curve, ROOT / "figures" / "nih_densenet_pca_learning_curve")
    print(json.dumps({
        "branch": branch,
        "intercept_only_at_72": result["intercept_only_at_72"],
        "coordinate_beats_mean": coordinate_hit,
        "prefix_beats_mean": prefix_hit,
        "coordinate": final["coordinate"]["mse_improvement_over_mean"],
        "prefix": final["prefix"]["mse_improvement_over_mean"],
        "oracle_prefix": final["oracle_prefix"]["mse_improvement_over_mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
