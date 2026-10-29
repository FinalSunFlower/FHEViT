"""Redraw the rank-1999 bridge on five training sets of 2000.

EVALUATION_PROTOCOL.md section 2.30. The published hash prefix is left in
place. The reserved confirmation images are not downloaded.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.nih_pa_only_power import _difference_sd  # noqa: E402
from experiments.validate_nih_oracle_auc import _log_odds  # noqa: E402
from experiments.validate_nih_unlabeled_bridge import (  # noqa: E402
    ANALYSIS_CACHE,
    CHUNK,
    POOL,
    TRAIN_CACHE,
    WIDTH,
    _auc_block,
    pca_from_gram,
)
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402
from experiments.validate_pneumonia_log_odds import NEGATIVE, POSITIVE, single_token_id  # noqa: E402
from fpavlm.evaluation import roc_auc  # noqa: E402

BACKEND = "internvl2-2b"
RANK = 1999
HOLDOUT = ROOT / "data" / "nih_chestxray14_final_holdout" / "positives.json"
SEALED = ROOT / "data" / "nih_chestxray14_sealed_confirmation"
STORED = ROOT / "experiments" / "results" / "nih_sealed_confirmation_auc_internvl2-2b.json"
STRATA = ROOT / "experiments" / "results" / "nih_sealed_strata.json"
DESTINATION = ROOT / "experiments" / "results" / "nih_rank1999_redraws_internvl2-2b.json"


def training_draws(pool_count, train_count, draws, seed):
    """Five locked calls of choice, without replacement, seed 42 in the run."""
    if train_count >= pool_count or draws < 1:
        raise ValueError("each draw must be a strict subset")
    generator = np.random.Generator(np.random.PCG64(seed))
    return [generator.choice(pool_count, size=train_count, replace=False) for _ in range(draws)]


def _column_block(cache, rows, start, stop):
    if rows is None:
        return np.asarray(cache[:, start:stop], dtype=np.float64)
    return np.asarray(cache[rows, start:stop], dtype=np.float64)


def _gram(cache, rows):
    width = cache.shape[1]
    mean = np.zeros(width, dtype=np.float64)
    for start in range(0, width, CHUNK):
        block = _column_block(cache, rows, start, start + CHUNK)
        mean[start:start + block.shape[1]] = block.mean(axis=0)
    count = cache.shape[0] if rows is None else len(rows)
    gram = np.zeros((count, count), dtype=np.float64)
    for start in range(0, width, CHUNK):
        block = _column_block(cache, rows, start, start + CHUNK)
        block -= mean[start:start + block.shape[1]]
        gram += block @ block.T
    return mean, gram


def _project(train_cache, rows, test_cache, mean, gram, coeff):
    train_scores = gram @ coeff
    test_scores = np.zeros((test_cache.shape[0], coeff.shape[1]), dtype=np.float64)
    width = train_cache.shape[1]
    for start in range(0, width, CHUNK):
        train_block = _column_block(train_cache, rows, start, start + CHUNK)
        test_block = _column_block(test_cache, None, start, start + CHUNK)
        train_block -= mean[start:start + train_block.shape[1]]
        test_block -= mean[start:start + test_block.shape[1]]
        test_scores += test_block @ (train_block.T @ coeff)
    return train_scores, test_scores


def _bridge(train_code, train_cache, rows, test_code):
    features = torch.as_tensor(np.asarray(train_code), dtype=torch.float64)
    feature_mean = features.mean(0)
    centered = features - feature_mean
    gram = centered.T @ centered + torch.eye(features.shape[1], dtype=torch.float64)
    coef = torch.linalg.solve(gram, centered.T)
    predicted = torch.empty(len(test_code), WIDTH, dtype=torch.float32)
    test = torch.as_tensor(np.asarray(test_code), dtype=torch.float32)
    for start in range(0, WIDTH, 4096):
        block = torch.as_tensor(_column_block(train_cache, rows, start, start + 4096))
        target_mean = block.mean(0)
        weight64 = coef @ (block - target_mean)
        bias = (target_mean - feature_mean.double() @ weight64).float()
        predicted[:, start:start + block.shape[1]] = test @ weight64.float() + bias
    return predicted.reshape(len(test_code), 256, 2048)


def _write(payload):
    DESTINATION.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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
    progress = json.loads(TRAIN_CACHE.with_suffix(".json").read_text(encoding="utf-8"))
    if progress["done"] != 3000 or progress["width"] != WIDTH:
        raise SystemExit("the unlabeled token cache is not the first 3000")
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
    draws = training_draws(3000, 2000, 5, 42)
    train_ids = [str(row["patient_id"]) for row in pool["records"][:3000]]
    train_cache = np.memmap(TRAIN_CACHE, dtype=np.float32, mode="r", shape=(3000, WIDTH))
    analysis_cache = np.memmap(ANALYSIS_CACHE, dtype=np.float32, mode="r", shape=(506, WIDTH))
    llm = load_backend(BACKEND)
    positive_id = single_token_id(llm.tokenizer, POSITIVE)
    negative_id = single_token_id(llm.tokenizer, NEGATIVE)
    cells = []
    payload = {
        "schema": "nih_rank1999_redraws_v1",
        "protocol_section": "2.30",
        "backend": BACKEND,
        "rank": RANK,
        "draw_count": 5,
        "train_count": 2000,
        "pool_count": 3000,
        "seed": 42,
        "published_prefix_replaced": False,
        "confirmation_opened": False,
        "reserved_images_downloaded": False,
        "cells": cells,
    }
    for number, rows in enumerate(draws):
        mean, gram = _gram(train_cache, rows)
        coeff = pca_from_gram(gram, RANK)
        train_code, test_code = _project(train_cache, rows, analysis_cache, mean, gram, coeff)
        predicted = _bridge(train_code, train_cache, rows, test_code)
        release = [
            _log_odds(llm, predicted[offset].unsqueeze(0), positive_id, negative_id)
            for offset in range(len(analysis))
        ]
        pa_only = _auc_block(
            [labels[offset] for offset in pa],
            [native_scores[offset] for offset in pa],
            [release[offset] for offset in pa],
            [analysis_ids[offset] for offset in pa],
        )
        cell = {
            "draw": number,
            "patient_ids": [train_ids[int(index)] for index in rows],
            "pa_only": pa_only,
            "passes": bool(pa_only["noninferior"]),
            "bootstrap_sd": _difference_sd(
                [labels[offset] for offset in pa],
                [native_scores[offset] for offset in pa],
                [release[offset] for offset in pa],
                [analysis_ids[offset] for offset in pa],
            ),
        }
        cells.append(cell)
        failures = sum(not item["passes"] for item in cells)
        payload["completed_draws"] = number + 1
        payload["failures"] = failures
        payload["unstable"] = failures >= 3
        payload["elapsed_s"] = time.perf_counter() - started
        _write(payload)
        print(json.dumps({
            "draw": number,
            "passes": cell["passes"],
            "difference": pa_only["difference"],
            "lower": pa_only["one_sided_95_lower"],
        }), flush=True)
        del predicted
    payload["complete"] = True
    payload["elapsed_s"] = time.perf_counter() - started
    _write(payload)
    print(json.dumps({
        "complete": True,
        "failures": payload["failures"],
        "unstable": payload["unstable"],
    }), flush=True)


if __name__ == "__main__":
    main()
