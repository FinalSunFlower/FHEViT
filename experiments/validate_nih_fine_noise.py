"""Fine noise budget at ranks 1024 and 1999.

EVALUATION_PROTOCOL.md section 2.37. Scales 0.05 and 0.1 only. The scales
already scored in the frontier are not rerun. The reserved confirmation
set is not used.
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
from experiments.validate_nih_coordinate_frontier import _penalized_bridge  # noqa: E402
from experiments.validate_nih_oracle_auc import _log_odds  # noqa: E402
from experiments.validate_nih_rank1999_redraws import (  # noqa: E402
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
    pca_from_gram,
)
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402
from experiments.validate_pneumonia_log_odds import NEGATIVE, POSITIVE, single_token_id  # noqa: E402
from fpavlm.evaluation import roc_auc  # noqa: E402

BACKEND = "internvl2-2b"
RANKS = (1024, 1999)
SCALES = (0.05, 0.1)
FRONTIER = ROOT / "experiments" / "results" / "nih_coordinate_frontier_internvl2-2b.json"
HOLDOUT = ROOT / "data" / "nih_chestxray14_final_holdout" / "positives.json"
SEALED = ROOT / "data" / "nih_chestxray14_sealed_confirmation"
STORED = ROOT / "experiments" / "results" / "nih_sealed_confirmation_auc_internvl2-2b.json"
STRATA = ROOT / "experiments" / "results" / "nih_sealed_strata.json"
DESTINATION = ROOT / "experiments" / "results" / f"nih_fine_noise_{BACKEND}.json"


def _published_zero():
    payload = json.loads(FRONTIER.read_text(encoding="utf-8"))
    if not payload.get("complete"):
        raise SystemExit("the scored frontier is not complete")
    found = {}
    for item in payload["summaries"]:
        if item["scale"] == 0.0 and item["rank"] in RANKS:
            found[item["rank"]] = item
    if set(found) != set(RANKS):
        raise SystemExit("the frontier is missing a zero-noise summary")
    return found


def _budget(rank, zero_passes, summaries):
    """Largest scale in {0, 0.05, 0.1} that still has at least 4 of 5 passes."""
    order = [(0.0, zero_passes)]
    for scale in SCALES:
        match = next(item for item in summaries if item["rank"] == rank and item["scale"] == scale)
        order.append((scale, match["passes"]))
    largest = None
    for scale, passed in order:
        if passed:
            largest = scale
        else:
            break
    return largest


def _write(payload):
    DESTINATION.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    torch.set_num_threads(4)
    torch.manual_seed(812)
    started = time.perf_counter()
    pool = json.loads((POOL / "metadata.json").read_text(encoding="utf-8"))
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    if holdout.get("images_downloaded") or holdout.get("auc_computed") or holdout.get("negative_sample_drawn"):
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
            raise SystemExit("a fine-noise draw does not match the stored five")
    zero = _published_zero()
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
    payload = {
        "schema": "nih_fine_noise_v1",
        "protocol_section": "2.37",
        "backend": BACKEND,
        "ranks": list(RANKS),
        "scales": list(SCALES),
        "same_five_draws": True,
        "noise_stream": "PCG64 seed 812, independent of the frontier stream",
        "not_differential_privacy": True,
        "confirmation_opened": False,
        "reserved_images_downloaded": False,
        "zero_noise_from_frontier": {
            str(rank): {"refit_passes": item["refit_passes"], "passes": item["passes"]}
            for rank, item in zero.items()
        },
        "cells": [],
        "complete": False,
    }
    done = set()
    if DESTINATION.is_file():
        existing = json.loads(DESTINATION.read_text(encoding="utf-8"))
        if existing.get("complete") is True:
            print("fine noise already complete", flush=True)
            return
        payload["cells"] = existing.get("cells", [])
        done = {(item["draw"], item["rank"], item["scale"]) for item in payload["cells"]}
    train_cache = np.memmap(TRAIN_CACHE, dtype=np.float32, mode="r", shape=(3000, WIDTH))
    analysis_cache = np.memmap(ANALYSIS_CACHE, dtype=np.float32, mode="r", shape=(506, WIDTH))
    llm = load_backend(BACKEND)
    positive_id = single_token_id(llm.tokenizer, POSITIVE)
    negative_id = single_token_id(llm.tokenizer, NEGATIVE)
    generator = np.random.Generator(np.random.PCG64(812))
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
                if (number, rank, scale) in done:
                    continue
                noise_std = scale * std
                noisy_train = train_code + train_noise * noise_std
                noisy_test = test_code + test_noise * noise_std
                predicted = _penalized_bridge(
                    noisy_train, train_cache, rows, noisy_test, noise_std ** 2,
                )
                release = [
                    _log_odds(llm, predicted[offset].unsqueeze(0), positive_id, negative_id)
                    for offset in range(predicted.shape[0])
                ]
                refit = _auc_block(
                    [labels[offset] for offset in pa],
                    [native_scores[offset] for offset in pa],
                    [release[offset] for offset in pa],
                    [analysis_ids[offset] for offset in pa],
                )
                cell = {
                    "draw": number,
                    "rank": rank,
                    "scale": scale,
                    "refit": refit,
                    "refit_passes": bool(refit["noninferior"]),
                }
                payload["cells"].append(cell)
                payload["elapsed_s"] = round(time.perf_counter() - started, 1)
                _write(payload)
                print(json.dumps({
                    "draw": number,
                    "rank": rank,
                    "scale": scale,
                    "difference": refit["difference"],
                    "lower": refit["one_sided_95_lower"],
                    "passes": cell["refit_passes"],
                }), flush=True)
                del predicted, release
                gc.collect()
        del full
    summaries = []
    for rank in RANKS:
        for scale in SCALES:
            passes = [
                item["refit_passes"] for item in payload["cells"]
                if item["rank"] == rank and item["scale"] == scale
            ]
            if len(passes) != 5:
                raise SystemExit(f"rank {rank} scale {scale} does not have five draws")
            summaries.append({
                "rank": rank,
                "scale": scale,
                "refit_passes": int(sum(passes)),
                "draws": 5,
                "passes": sum(passes) >= 4,
            })
    payload["summaries"] = summaries
    payload["error_budget"] = {
        str(rank): _budget(rank, zero[rank]["passes"], summaries) for rank in RANKS
    }
    payload["complete"] = True
    payload["confirmation_opened"] = False
    payload["reserved_images_downloaded"] = False
    payload["elapsed_s"] = round(time.perf_counter() - started, 1)
    _write(payload)
    print(json.dumps({"complete": True, "error_budget": payload["error_budget"]}), flush=True)


if __name__ == "__main__":
    main()
