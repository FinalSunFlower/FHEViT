"""Depth-1 and depth-2 pixel students, plus the layer-1 controls.

EVALUATION_PROTOCOL.md sections 2.35 and 2.38. The students do not refit the
oracle bridge. The reserved confirmation set is not used.
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

from experiments.nih_redraw_reading import REDRAWS  # noqa: E402
from experiments.validate_nih_oracle_auc import _log_odds  # noqa: E402
from experiments.validate_nih_passing_release_attack import _images  # noqa: E402
from experiments.validate_nih_rank1999_redraws import _bridge, _gram, _project, training_draws  # noqa: E402
from experiments.validate_nih_unlabeled_bridge import (  # noqa: E402
    POOL,
    SEALED,
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
STORED = ROOT / "experiments" / "results" / "nih_sealed_confirmation_auc_internvl2-2b.json"
STRATA = ROOT / "experiments" / "results" / "nih_sealed_strata.json"
DESTINATION = ROOT / "experiments" / "results" / "nih_pixel_students_internvl2-2b.json"
LAYER1_TRAIN = POOL / "layer1_mean.float32"
LAYER1_ANALYSIS = POOL / "analysis_layer1_mean.float32"
CIRCUIT = ("pixel_linear", "pixel_square")
CONTROLS = ("layer1_linear", "layer1_square")


def coordinate_map(features, targets):
    """Ridge penalty 1 and sigma 0 on the centered unnormalized gram."""
    source = torch.as_tensor(np.asarray(features), dtype=torch.float64)
    outcome = torch.as_tensor(np.asarray(targets), dtype=torch.float64)
    source_mean = source.mean(0)
    outcome_mean = outcome.mean(0)
    centered = source - source_mean
    gram = centered.T @ centered + torch.eye(centered.shape[1], dtype=torch.float64)
    weight = torch.linalg.solve(gram, centered.T @ (outcome - outcome_mean))
    return source_mean, outcome_mean, weight


def apply_map(features, source_mean, outcome_mean, weight):
    centered = torch.as_tensor(np.asarray(features), dtype=torch.float64) - source_mean
    return (centered @ weight + outcome_mean).numpy()


def _square(features):
    values = np.asarray(features, dtype=np.float32)
    return np.concatenate([values, values * values], axis=1)


def _pixels(root, records):
    return _images(root, records).reshape(len(records), -1)


def _layer1(llm, root, records, path, width):
    if path.is_file() and path.stat().st_size == len(records) * width * 4:
        return np.memmap(path, dtype=np.float32, mode="r", shape=(len(records), width))
    cache = np.memmap(path, dtype=np.float32, mode="w+", shape=(len(records), width))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    from PIL import Image

    vision = llm.model.vision_model
    for index, row in enumerate(records):
        with Image.open(root / row["relative_path"]) as image:
            array = np.asarray(image.convert("RGB").resize((448, 448), Image.Resampling.BILINEAR), dtype=np.float32)
        array = (array / 255.0 - mean) / std
        pixels = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            output = vision(
                pixel_values=pixels.to("cuda", dtype=torch.bfloat16),
                output_hidden_states=True,
                return_dict=True,
            )
            if output.hidden_states is None or len(output.hidden_states) < 2:
                raise RuntimeError("the first InternViT block did not return its tokens")
            cache[index] = output.hidden_states[1].mean(dim=1).float().cpu().numpy()[0]
        if index % 100 == 0:
            print(f"layer1 {path.name} {index}/{len(records)}", flush=True)
    cache.flush()
    return cache


def _write(payload):
    DESTINATION.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _score_block(llm, predicted, labels, native_scores, analysis_ids, pa, positive_id, negative_id):
    release = [
        _log_odds(llm, predicted[offset].unsqueeze(0), positive_id, negative_id)
        for offset in range(predicted.shape[0])
    ]
    return _auc_block(
        [labels[offset] for offset in pa],
        [native_scores[offset] for offset in pa],
        [release[offset] for offset in pa],
        [analysis_ids[offset] for offset in pa],
    )


def main():
    torch.set_num_threads(4)
    started = time.perf_counter()
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    if holdout.get("images_downloaded") or holdout.get("auc_computed") or holdout.get("negative_sample_drawn"):
        raise SystemExit("the reserved confirmation pool was already opened")
    pool = json.loads((POOL / "metadata.json").read_text(encoding="utf-8"))
    if pool.get("download_prefix") != 3000 or not pool.get("download_prefix_complete"):
        raise SystemExit("the cached prefix is not the completed 3000")
    redraws = json.loads(REDRAWS.read_text(encoding="utf-8"))
    train_ids = [str(row["patient_id"]) for row in pool["records"][:3000]]
    draws = training_draws(3000, 2000, 5, 42)
    for draw, cell in zip(draws, redraws["cells"]):
        if [train_ids[int(index)] for index in draw] != cell["patient_ids"]:
            raise SystemExit("a student draw does not match the stored five")
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
    if abs(roc_auc([labels[offset] for offset in pa], [native_scores[offset] for offset in pa]) - stored_pa) > 1e-6:
        raise SystemExit("the stored PA-only native AUC disagrees with section 2.23")
    payload = {
        "schema": "nih_pixel_students_v1",
        "protocol_section": "2.35",
        "precheck_section": "2.38",
        "backend": BACKEND,
        "rank": RANK,
        "confirmation_opened": False,
        "reserved_images_downloaded": False,
        "cells": [],
        "complete": False,
    }
    done = set()
    if DESTINATION.is_file():
        existing = json.loads(DESTINATION.read_text(encoding="utf-8"))
        if existing.get("complete") is True:
            print("pixel students already complete", flush=True)
            return
        payload["cells"] = existing.get("cells", [])
        done = {(item["draw"], item["name"]) for item in payload["cells"]}
    train_records = pool["records"][:3000]
    train_pixels = _pixels(POOL, train_records)
    test_pixels = _pixels(SEALED, analysis)
    train_cache = np.memmap(TRAIN_CACHE, dtype=np.float32, mode="r", shape=(3000, WIDTH))
    print("loading the backend", flush=True)
    llm = load_backend(BACKEND)
    positive_id = single_token_id(llm.tokenizer, POSITIVE)
    negative_id = single_token_id(llm.tokenizer, NEGATIVE)
    layer1_train = _layer1(llm, POOL, train_records, LAYER1_TRAIN, 1024)
    layer1_test = _layer1(llm, SEALED, analysis, LAYER1_ANALYSIS, 1024)
    feature_bank = {
        "pixel_linear": (train_pixels, test_pixels),
        "pixel_square": (_square(train_pixels), _square(test_pixels)),
        "layer1_linear": (np.asarray(layer1_train), np.asarray(layer1_test)),
        "layer1_square": (_square(layer1_train), _square(layer1_test)),
    }
    for number, rows in enumerate(draws):
        mean, gram = _gram(train_cache, rows)
        coeff = pca_from_gram(gram, RANK)
        train_code, targets = _project(train_cache, rows, train_cache, mean, gram, coeff)
        if targets.shape != (3000, RANK):
            raise SystemExit("the projected training targets are not rank 1999")
        for name in (*CIRCUIT, *CONTROLS):
            if (number, name) in done:
                continue
            source_mean, outcome_mean, weight = coordinate_map(feature_bank[name][0], targets)
            predicted_code = apply_map(feature_bank[name][1], source_mean, outcome_mean, weight)
            predicted = _bridge(train_code, train_cache, rows, predicted_code)
            block = _score_block(
                llm, predicted, labels, native_scores, analysis_ids, pa, positive_id, negative_id,
            )
            cell = {
                "draw": number,
                "name": name,
                "circuit_candidate": name in CIRCUIT,
                "pa_only": block,
                "passes": bool(block["noninferior"]),
            }
            payload["cells"].append(cell)
            payload["elapsed_s"] = round(time.perf_counter() - started, 1)
            _write(payload)
            print(json.dumps({
                "draw": number,
                "name": name,
                "difference": block["difference"],
                "lower": block["one_sided_95_lower"],
                "passes": cell["passes"],
            }), flush=True)
            del predicted
    summaries = []
    for name in (*CIRCUIT, *CONTROLS):
        passes = [item["passes"] for item in payload["cells"] if item["name"] == name]
        if len(passes) != 5:
            raise SystemExit(f"{name} does not have five draws")
        summaries.append({
            "name": name,
            "circuit_candidate": name in CIRCUIT,
            "refit_passes": int(sum(passes)),
            "draws": 5,
            "passes": sum(passes) >= 4,
        })
    payload["summaries"] = summaries
    payload["no_feasible_student"] = not any(item["passes"] for item in summaries if item["circuit_candidate"])
    payload["complete"] = True
    payload["confirmation_opened"] = False
    payload["reserved_images_downloaded"] = False
    payload["elapsed_s"] = round(time.perf_counter() - started, 1)
    _write(payload)
    print(json.dumps({
        "complete": True,
        "no_feasible_student": payload["no_feasible_student"],
        "summaries": summaries,
    }), flush=True)


if __name__ == "__main__":
    main()
