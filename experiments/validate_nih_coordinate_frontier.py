"""Score the coordinate frontier on the five stored training draws.

EVALUATION_PROTOCOL.md section 2.34. Scale 0 on the published first 2000
patients must match the published PA difference before any other cell is
kept. The reserved confirmation is not downloaded. Reconstructors are not
trained in this file.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.nih_redraw_reading import REDRAWS  # noqa: E402
from experiments.validate_nih_oracle_auc import _log_odds  # noqa: E402
from experiments.validate_nih_rank1999_redraws import (  # noqa: E402
    _bridge,
    _gram,
    _project,
    training_draws,
)
from experiments.validate_nih_unlabeled_bridge import (  # noqa: E402
    ANALYSIS_CACHE,
    POOL,
    TRAIN_CACHE,
    WIDTH,
    _auc_block,
    bridge_from_rows,
    gram_from_rows,
    pca_from_gram,
    project_rows,
)
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402
from experiments.validate_pneumonia_log_odds import NEGATIVE, POSITIVE, single_token_id  # noqa: E402
from fpavlm.evaluation import roc_auc  # noqa: E402

BACKEND = "internvl2-2b"
RANKS = (256, 512, 1024, 1999)
SCALES = (0.0, 0.25, 0.5, 1.0)
TOLERANCE = 1e-6
HOLDOUT = ROOT / "data" / "nih_chestxray14_final_holdout" / "positives.json"
SEALED = ROOT / "data" / "nih_chestxray14_sealed_confirmation"
STORED = ROOT / "experiments" / "results" / "nih_sealed_confirmation_auc_internvl2-2b.json"
STRATA = ROOT / "experiments" / "results" / "nih_sealed_strata.json"
PUBLISHED = ROOT / "experiments" / "results" / "nih_unlabeled_bridge_internvl2-2b.json"
DESTINATION = ROOT / "experiments" / "results" / "nih_coordinate_frontier_internvl2-2b.json"


def feature_gram(centered, variance):
    """Unnormalized gram plus ``1 + n * v`` on each coordinate."""
    variance = torch.as_tensor(variance, dtype=centered.dtype, device=centered.device)
    if variance.ndim != 1 or variance.shape[0] != centered.shape[1]:
        raise ValueError("variance must match the coordinate count")
    if torch.any(variance < 0):
        raise ValueError("variance must be nonnegative")
    gram = centered.T @ centered
    gram.diagonal().add_(1 + centered.shape[0] * variance)
    return gram


def _penalized_bridge(train_code, train_cache, rows, test_code, variance):
    if np.all(np.asarray(variance) == 0):
        return _bridge(train_code, train_cache, rows, test_code)
    features = torch.as_tensor(np.asarray(train_code), dtype=torch.float64)
    feature_mean = features.mean(0)
    centered = features - feature_mean
    gram = feature_gram(centered, torch.as_tensor(variance, dtype=torch.float64))
    coef = torch.linalg.solve(gram, centered.T)
    predicted = torch.empty(len(test_code), WIDTH, dtype=torch.float32)
    test = torch.as_tensor(np.asarray(test_code), dtype=torch.float32)
    for start in range(0, WIDTH, 4096):
        block = torch.as_tensor(np.asarray(train_cache[rows, start:start + 4096], dtype=np.float64))
        target_mean = block.mean(0)
        weight64 = coef @ (block - target_mean)
        bias = (target_mean - feature_mean.double() @ weight64).float()
        predicted[:, start:start + block.shape[1]] = test @ weight64.float() + bias
    return predicted.reshape(len(test_code), 256, 2048)


def _scores(llm, predicted, positive_id, negative_id):
    return [
        _log_odds(llm, predicted[offset].unsqueeze(0), positive_id, negative_id)
        for offset in range(predicted.shape[0])
    ]


def _pa(labels, native, release, patients, pa):
    return _auc_block(
        [labels[offset] for offset in pa],
        [native[offset] for offset in pa],
        [release[offset] for offset in pa],
        [patients[offset] for offset in pa],
    )


def _write(payload):
    DESTINATION.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _published_difference():
    payload = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    cell = next(
        item for item in payload["cells"]
        if item["prefix"] == 2000 and item["rank"] == "n_train_minus_1"
    )
    return float(cell["pa_only"]["difference"])


def main():
    torch.set_num_threads(4)
    torch.manual_seed(812)
    started = time.perf_counter()
    pool = json.loads((POOL / "metadata.json").read_text(encoding="utf-8"))
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    if holdout.get("images_downloaded") or holdout.get("auc_computed"):
        raise SystemExit("the reserved confirmation pool was already opened")
    if pool.get("download_prefix") != 3000 or not pool.get("download_prefix_complete"):
        raise SystemExit("the cached prefix is not the completed 3000")
    redraws = json.loads(REDRAWS.read_text(encoding="utf-8"))
    if not redraws.get("complete") or len(redraws["cells"]) != 5:
        raise SystemExit("the five stored draws are not complete")
    train_ids = [str(row["patient_id"]) for row in pool["records"][:3000]]
    draws = training_draws(3000, 2000, 5, 42)
    for draw, cell in zip(draws, redraws["cells"]):
        recovered = [train_ids[int(index)] for index in draw]
        if recovered != cell["patient_ids"]:
            raise SystemExit("a frontier draw does not match the stored five")
    sealed = json.loads((SEALED / "metadata.json").read_text(encoding="utf-8"))
    analysis = sealed["records"]
    stored = json.loads(STORED.read_text(encoding="utf-8"))["ranks"][0]["reports"]
    analysis_ids = [str(row["patient_id"]) for row in analysis]
    if [row["patient_id"] for row in stored] != analysis_ids:
        raise SystemExit("stored native scores do not follow the analysis manifest")
    native_scores = [float(row["native_log_odds"]) for row in stored]
    labels = [int(row["label_pneumonia"]) for row in analysis]
    pa = [offset for offset, row in enumerate(analysis) if row["view_position"] == "PA"]
    stored_pa = json.loads(STRATA.read_text(encoding="utf-8"))["pa_only"]["native"]["auc"]
    pa_auc = roc_auc([labels[offset] for offset in pa], [native_scores[offset] for offset in pa])
    if abs(pa_auc - stored_pa) > 1e-6:
        raise SystemExit("the stored PA-only native AUC disagrees with section 2.23")
    target = _published_difference()
    train_cache = np.memmap(TRAIN_CACHE, dtype=np.float32, mode="r", shape=(3000, WIDTH))
    analysis_cache = np.memmap(ANALYSIS_CACHE, dtype=np.float32, mode="r", shape=(506, WIDTH))
    llm = load_backend(BACKEND)
    positive_id = single_token_id(llm.tokenizer, POSITIVE)
    negative_id = single_token_id(llm.tokenizer, NEGATIVE)
    prefix = train_cache[:2000]
    mean, gram = gram_from_rows(prefix)
    coeff = pca_from_gram(gram, 1999)
    train_code, test_code = project_rows(prefix, analysis_cache, mean, gram, coeff)
    predicted = bridge_from_rows(train_code, prefix, test_code)
    release = _scores(llm, predicted, positive_id, negative_id)
    regression = _pa(labels, native_scores, release, analysis_ids, pa)
    gap = abs(regression["difference"] - target)
    payload = {
        "schema": "nih_coordinate_frontier_v1",
        "protocol_section": "2.34",
        "backend": BACKEND,
        "ranks": list(RANKS),
        "scales": list(SCALES),
        "same_five_draws": True,
        "attacker_scored": False,
        "confirmation_opened": False,
        "reserved_images_downloaded": False,
        "regression": {
            "prefix": 2000,
            "rank": 1999,
            "scale": 0.0,
            "published_difference": target,
            "difference": regression["difference"],
            "absolute_gap": gap,
            "tolerance": TOLERANCE,
            "matches": gap <= TOLERANCE,
            "pa_only": regression,
        },
        "cells": [],
    }
    payload["elapsed_s"] = time.perf_counter() - started
    _write(payload)
    print(json.dumps({
        "regression_difference": regression["difference"],
        "published_difference": target,
        "matches": payload["regression"]["matches"],
    }), flush=True)
    del predicted
    if not payload["regression"]["matches"]:
        raise SystemExit("scale 0 does not reproduce the published PA difference")
    generator = np.random.Generator(np.random.PCG64(812))
    grouped = {}
    for number, rows in enumerate(draws):
        mean, gram = _gram(train_cache, rows)
        full = pca_from_gram(gram, 1999)
        for rank in RANKS:
            train_code, test_code = _project(
                train_cache, rows, analysis_cache, mean, gram, full[:, :rank],
            )
            std = np.std(train_code, axis=0, ddof=0)
            for scale in SCALES:
                train_noise = generator.standard_normal(train_code.shape)
                test_noise = generator.standard_normal(test_code.shape)
                noise_std = scale * std
                if scale == 0:
                    refit_predicted = _bridge(train_code, train_cache, rows, test_code)
                    mismatch_predicted = refit_predicted
                else:
                    noisy_train = train_code + train_noise * noise_std
                    noisy_test = test_code + test_noise * noise_std
                    refit_predicted = _penalized_bridge(
                        noisy_train, train_cache, rows, noisy_test, noise_std ** 2,
                    )
                    mismatch_predicted = _bridge(train_code, train_cache, rows, noisy_test)
                refit_release = _scores(llm, refit_predicted, positive_id, negative_id)
                refit = _pa(labels, native_scores, refit_release, analysis_ids, pa)
                if scale == 0:
                    mismatch = refit
                else:
                    mismatch_release = _scores(llm, mismatch_predicted, positive_id, negative_id)
                    mismatch = _pa(labels, native_scores, mismatch_release, analysis_ids, pa)
                    del mismatch_release
                cell = {
                    "draw": number,
                    "rank": rank,
                    "scale": scale,
                    "refit": refit,
                    "mismatch": mismatch,
                    "refit_passes": bool(refit["noninferior"]),
                }
                payload["cells"].append(cell)
                key = (rank, scale)
                grouped.setdefault(key, []).append(cell["refit_passes"])
                payload["elapsed_s"] = time.perf_counter() - started
                _write(payload)
                print(json.dumps({
                    "draw": number,
                    "rank": rank,
                    "scale": scale,
                    "refit_difference": refit["difference"],
                    "refit_lower": refit["one_sided_95_lower"],
                    "refit_passes": cell["refit_passes"],
                }), flush=True)
                del refit_predicted, mismatch_predicted
                gc.collect()
        del full
    payload["summaries"] = []
    for rank, scale in ((rank, scale) for rank in RANKS for scale in SCALES):
        passes = grouped[(rank, scale)]
        payload["summaries"].append({
            "rank": rank,
            "scale": scale,
            "refit_passes": sum(passes),
            "draws": len(passes),
            "passes": sum(passes) >= 4,
        })
    payload["complete"] = True
    payload["attacker_scored"] = False
    payload["confirmation_opened"] = False
    payload["reserved_images_downloaded"] = False
    payload["elapsed_s"] = time.perf_counter() - started
    _write(payload)
    print(json.dumps({
        "complete": True,
        "passing": [item for item in payload["summaries"] if item["passes"]],
    }), flush=True)


if __name__ == "__main__":
    main()
