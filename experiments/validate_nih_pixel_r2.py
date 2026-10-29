"""Unconstrained pixel ridge as a pre-check, not a student gate.

EVALUATION_PROTOCOL.md section 2.38. Fit on the 3000. Score the 506.
The depth-1 and depth-2 students still run after this file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.validate_nih_passing_release_attack import (  # noqa: E402
    ANALYSIS_COUNT,
    BASIS_COUNT,
    RANK,
    TRAIN_COUNT,
    WIDTH,
    _coordinates,
    _images,
)
from experiments.validate_nih_unlabeled_bridge import ANALYSIS_CACHE, POOL, SEALED  # noqa: E402

SEGMENTS = ((0, 64), (64, 256), (256, 1024), (1024, 1999))
HOLDOUT = ROOT / "data" / "nih_chestxray14_final_holdout" / "positives.json"
OUTPUT = ROOT / "experiments" / "results" / "nih_pixel_r2_internvl2-2b.json"


def _ridge(pixels, targets):
    features = torch.as_tensor(pixels, dtype=torch.float64)
    outcomes = torch.as_tensor(targets, dtype=torch.float64)
    feature_mean = features.mean(0)
    target_mean = outcomes.mean(0)
    centered = features - feature_mean
    gram = centered.T @ centered + torch.eye(centered.shape[1], dtype=torch.float64)
    weight = torch.linalg.solve(gram, centered.T @ (outcomes - target_mean))
    return feature_mean, target_mean, weight


def _predict(pixels, feature_mean, target_mean, weight):
    centered = torch.as_tensor(pixels, dtype=torch.float64) - feature_mean
    return (centered @ weight + target_mean).numpy()


def _segment(start, stop, truth, predicted, train_codes):
    gold = truth[:, start:stop]
    forecast = predicted[:, start:stop]
    baseline = train_codes[:, start:stop].mean(axis=0)
    residual = gold - forecast
    total = gold - baseline
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum(total ** 2))
    if ss_tot == 0.0:
        raise ValueError("a coordinate segment has no training variation on the 506")
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    std = np.std(train_codes[:, start:stop], axis=0, ddof=0)
    segment_std = float(np.sqrt(np.mean(std ** 2)))
    threshold = 0.25 * segment_std
    return {
        "start": start,
        "stop": stop,
        "r2": 1.0 - ss_res / ss_tot,
        "rmse": rmse,
        "training_std": segment_std,
        "noise_0_25_std": threshold,
        "far_from_0_25": rmse > threshold,
    }


def main():
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    if holdout.get("images_downloaded") or holdout.get("auc_computed") or holdout.get("negative_sample_drawn"):
        raise SystemExit("the reserved confirmation pool was already opened")
    pool = json.loads((POOL / "metadata.json").read_text(encoding="utf-8"))
    sealed = json.loads((SEALED / "metadata.json").read_text(encoding="utf-8"))
    train_records = pool["records"][:TRAIN_COUNT]
    test_records = sealed["records"]
    if len(test_records) != ANALYSIS_COUNT:
        raise SystemExit("the analysis manifest is not the 506")
    train_cache = np.memmap(POOL / "token_cache.float32", dtype=np.float32, mode="r", shape=(TRAIN_COUNT, WIDTH))
    analysis_cache = np.memmap(ANALYSIS_CACHE, dtype=np.float32, mode="r", shape=(ANALYSIS_COUNT, WIDTH))
    train_codes, test_codes = _coordinates(train_cache, analysis_cache)
    if train_codes.shape != (TRAIN_COUNT, RANK) or test_codes.shape != (ANALYSIS_COUNT, RANK):
        raise SystemExit("the rank-1999 coordinates do not match section 2.36")
    del train_cache, analysis_cache
    train_pixels = _images(POOL, train_records).reshape(TRAIN_COUNT, -1)
    test_pixels = _images(SEALED, test_records).reshape(ANALYSIS_COUNT, -1)
    feature_mean, target_mean, weight = _ridge(train_pixels, train_codes)
    predicted = _predict(test_pixels, feature_mean, target_mean, weight)
    segments = [
        _segment(start, stop, test_codes, predicted, train_codes) for start, stop in SEGMENTS
    ]
    payload = {
        "schema": "nih_pixel_r2_v1",
        "protocol_section": "2.38",
        "basis_count": BASIS_COUNT,
        "train_count": TRAIN_COUNT,
        "analysis_count": ANALYSIS_COUNT,
        "rank": RANK,
        "penalty": 1,
        "heuristic": True,
        "cancels_student": False,
        "confirmation_opened": False,
        "segments": segments,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(segments, indent=2), flush=True)


if __name__ == "__main__":
    main()
