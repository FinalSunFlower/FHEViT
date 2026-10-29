"""Score the unlabeled PCA bridge on the prefixes whose images are on disk.

EVALUATION_PROTOCOL.md section 2.27. The locked sizes are 500, 2000, and
8000. This run scores a size only when that prefix's images exist. The
8000-patient attacker in section 2.28 stays unfitted. The reserve is not
downloaded.
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

from experiments.evaluate_diagnostic_prefix import extract  # noqa: E402
from experiments.nih_pa_only_power import _difference_sd  # noqa: E402
from experiments.nih_sample_size import power_noninferior, se_at_n  # noqa: E402
from experiments.validate_nih_oracle_auc import _log_odds  # noqa: E402
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402
from experiments.validate_pneumonia_log_odds import NEGATIVE, POSITIVE, single_token_id  # noqa: E402
from fpavlm.evaluation import paired_auc_noninferiority, roc_auc  # noqa: E402

BACKEND = "internvl2-2b"
LOCKED_PREFIXES = (500, 2000, 8000)
FIXED_RANKS = (64, 128, 256)
WIDTH = 256 * 2048
CHUNK = 8192
POOL = ROOT / "data" / "nih_chestxray14_unlabeled_pool"
SEALED = ROOT / "data" / "nih_chestxray14_sealed_confirmation"
HOLDOUT = ROOT / "data" / "nih_chestxray14_final_holdout" / "positives.json"
STORED = ROOT / "experiments" / "results" / "nih_sealed_confirmation_auc_internvl2-2b.json"
STRATA = ROOT / "experiments" / "results" / "nih_sealed_strata.json"
DESTINATION = ROOT / "experiments" / "results" / "nih_unlabeled_bridge_internvl2-2b.json"
TRAIN_CACHE = POOL / "token_cache.float32"
ANALYSIS_CACHE = POOL / "analysis_token_cache.float32"


def scheduled_prefixes(images_ready):
    """Locked sizes that are fully downloaded. 3000 is not a new grid size."""
    if images_ready < 1:
        raise ValueError("at least one image is required")
    return tuple(size for size in LOCKED_PREFIXES if size <= images_ready)


def rank_plan(n_train):
    """Fixed ranks that fit, then the training maximum n_train - 1."""
    if n_train < 3:
        raise ValueError("PCA needs at least three training patients")
    maximum = n_train - 1
    return tuple(rank for rank in FIXED_RANKS if rank < maximum) + (maximum,)


def gram_from_rows(rows):
    """Column-chunked Gram matrix. ``rows`` is addressable as (patients, width)."""
    count, width = rows.shape
    mean = np.zeros(width, dtype=np.float64)
    for start in range(0, width, CHUNK):
        block = np.asarray(rows[:, start:start + CHUNK], dtype=np.float64)
        mean[start:start + block.shape[1]] = block.mean(axis=0)
    gram = np.zeros((count, count), dtype=np.float64)
    for start in range(0, width, CHUNK):
        block = np.asarray(rows[:, start:start + CHUNK], dtype=np.float64)
        block -= mean[start:start + block.shape[1]]
        gram += block @ block.T
    return mean, gram


def pca_from_gram(gram, rank):
    values, vectors = torch.linalg.eigh(torch.from_numpy(np.asarray(gram, dtype=np.float64)))
    order = torch.argsort(values, descending=True)
    chosen = [int(index) for index in order.tolist() if float(values[index]) > 1e-8][:rank]
    if len(chosen) < rank:
        raise ValueError("PCA could not fill the requested rank")
    picked = torch.tensor(chosen)
    scale = values[picked].sqrt()
    return (vectors[:, picked] / scale).numpy()


def project_rows(train_rows, test_rows, mean, gram, coeff):
    """Project with the dual basis. ``coeff`` is (train_patients, rank)."""
    train_scores = gram @ coeff
    test_scores = np.zeros((test_rows.shape[0], coeff.shape[1]), dtype=np.float64)
    width = train_rows.shape[1]
    for start in range(0, width, CHUNK):
        train_block = np.asarray(train_rows[:, start:start + CHUNK], dtype=np.float64)
        test_block = np.asarray(test_rows[:, start:start + CHUNK], dtype=np.float64)
        train_block -= mean[start:start + train_block.shape[1]]
        test_block -= mean[start:start + test_block.shape[1]]
        test_scores += test_block @ (train_block.T @ coeff)
    return train_scores, test_scores


def bridge_from_rows(train_code, train_rows, test_code):
    """Same ridge-1 product as the token bridge, read in column chunks."""
    features = torch.as_tensor(np.asarray(train_code), dtype=torch.float64)
    feature_mean = features.mean(0)
    centered = features - feature_mean
    gram = centered.T @ centered + torch.eye(features.shape[1], dtype=torch.float64)
    coef = torch.linalg.solve(gram, centered.T)
    width = train_rows.shape[1]
    predicted = torch.empty(len(test_code), width, dtype=torch.float32)
    test = torch.as_tensor(np.asarray(test_code), dtype=torch.float32)
    for start in range(0, width, 4096):
        block = torch.as_tensor(np.asarray(train_rows[:, start:start + 4096], dtype=np.float64))
        target_mean = block.mean(0)
        weight64 = coef @ (block - target_mean)
        weight = weight64.float()
        bias = (target_mean - feature_mean.double() @ weight64).float()
        predicted[:, start:start + block.shape[1]] = test @ weight + bias
    if predicted.shape[1] == WIDTH:
        return predicted.reshape(len(test_code), 256, 2048)
    return predicted


def _open_cache(path, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.stat().st_size != count * WIDTH * 4:
        buffer = np.memmap(path, dtype=np.float32, mode="w+", shape=(count, WIDTH))
        buffer.flush()
        return buffer, 0
    return np.memmap(path, dtype=np.float32, mode="r+", shape=(count, WIDTH)), count


def _cache_progress(path):
    marker = path.with_suffix(".json")
    if not marker.is_file():
        return 0, None
    payload = json.loads(marker.read_text(encoding="utf-8"))
    return int(payload["done"]), payload["patient_ids"]


def _mark_progress(path, done, patient_ids):
    path.with_suffix(".json").write_text(json.dumps({
        "done": done, "patient_ids": patient_ids, "width": WIDTH,
    }) + "\n", encoding="utf-8")


def _extract_into(llm, rows, root, cache_path):
    patient_ids = [str(row["patient_id"]) for row in rows]
    done, stored_ids = _cache_progress(cache_path)
    if stored_ids not in (None, patient_ids) or done > len(rows):
        raise SystemExit(f"{cache_path.name} does not match this patient list")
    cache, existing = _open_cache(cache_path, len(rows))
    if existing == 0:
        done = 0
    for offset in range(done, len(rows)):
        row = rows[offset]
        tokens = extract(llm, root / row["relative_path"], BACKEND)
        flat = tokens.reshape(-1).numpy().astype(np.float32, copy=False)
        if flat.shape != (WIDTH,):
            raise SystemExit("native tokens are not 256 by 2048")
        cache[offset] = flat
        if offset % 20 == 0 or offset + 1 == len(rows):
            cache.flush()
            _mark_progress(cache_path, offset + 1, patient_ids)
            print(json.dumps({"cache": cache_path.name, "done": offset + 1, "of": len(rows)}), flush=True)
    cache.flush()
    _mark_progress(cache_path, len(rows), patient_ids)
    return cache


def _auc_block(labels, native, release, patients):
    return paired_auc_noninferiority(
        labels, native, release, patients, margin=0.05, bootstrap_samples=2000, seed=42,
    )


def main():
    torch.set_num_threads(4)
    torch.manual_seed(812)
    started = time.perf_counter()
    pool = json.loads((POOL / "metadata.json").read_text(encoding="utf-8"))
    if pool.get("confirmation_opened") or pool.get("auc_computed") or pool.get("attacker_scored"):
        raise SystemExit("the unlabeled pool was already used as a scored confirmation set")
    if pool.get("protocol_section") != "2.27" or len(pool["records"]) != 8000:
        raise SystemExit("the unlabeled list is not the frozen 8000")
    if not pool.get("download_prefix_complete") or pool.get("download_prefix") != 3000:
        raise SystemExit("the first 3000 images are not the completed download")
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    if holdout.get("images_downloaded") or holdout.get("auc_computed"):
        raise SystemExit("the reserved confirmation pool was already opened")
    sealed = json.loads((SEALED / "metadata.json").read_text(encoding="utf-8"))
    analysis = sealed["records"]
    if len(analysis) != 506 or any(row.get("download_error") for row in analysis):
        raise SystemExit("the 506 analysis images are not complete")
    ready = sum(1 for row in pool["records"][:3000] if (POOL / "images" / row["image_index"]).is_file())
    if ready != 3000:
        raise SystemExit("the first 3000 image files are not all on disk")
    prefixes = scheduled_prefixes(ready)
    if prefixes != (500, 2000):
        raise SystemExit("this disk state should score only the 500 and 2000 prefixes")
    reserved = {str(row["patient_id"]) for row in holdout["records"]}
    train_ids = [str(row["patient_id"]) for row in pool["records"][:3000]]
    analysis_ids = [str(row["patient_id"]) for row in analysis]
    if reserved & set(train_ids) or set(analysis_ids) & set(train_ids):
        raise SystemExit("an evaluation patient entered the unlabeled prefix")
    stored = json.loads(STORED.read_text(encoding="utf-8"))["ranks"][0]["reports"]
    if [row["patient_id"] for row in stored] != analysis_ids:
        raise SystemExit("stored native scores do not follow the analysis manifest")
    native_scores = [float(row["native_log_odds"]) for row in stored]
    labels = [int(row["label_pneumonia"]) for row in analysis]
    views = [row["view_position"] for row in analysis]
    pa = [offset for offset, view in enumerate(views) if view == "PA"]
    stored_pa = json.loads(STRATA.read_text(encoding="utf-8"))["pa_only"]["native"]["auc"]
    pa_auc = roc_auc([labels[offset] for offset in pa], [native_scores[offset] for offset in pa])
    if abs(pa_auc - stored_pa) > 1e-6:
        raise SystemExit("the stored PA-only native AUC disagrees with section 2.23")
    llm = load_backend(BACKEND)
    positive_id = single_token_id(llm.tokenizer, POSITIVE)
    negative_id = single_token_id(llm.tokenizer, NEGATIVE)
    train_cache = _extract_into(llm, pool["records"][:3000], POOL, TRAIN_CACHE)
    analysis_cache = _extract_into(llm, analysis, SEALED, ANALYSIS_CACHE)
    cells = []
    for size in prefixes:
        mean, gram = gram_from_rows(train_cache[:size])
        plan = rank_plan(size)
        coeff = pca_from_gram(gram, plan[-1])
        for rank in plan:
            used = coeff[:, :rank]
            train_code, test_code = project_rows(
                train_cache[:size], analysis_cache, mean, gram, used,
            )
            predicted = bridge_from_rows(train_code, train_cache[:size], test_code)
            release = [
                _log_odds(llm, predicted[offset].unsqueeze(0), positive_id, negative_id)
                for offset in range(len(analysis))
            ]
            name = "n_train_minus_1" if rank == size - 1 else str(rank)
            full = _auc_block(labels, native_scores, release, analysis_ids)
            pa_only = _auc_block(
                [labels[offset] for offset in pa],
                [native_scores[offset] for offset in pa],
                [release[offset] for offset in pa],
                [analysis_ids[offset] for offset in pa],
            )
            cell = {
                "prefix": size,
                "rank": name,
                "rank_order": rank,
                "full_set": full,
                "pa_only": pa_only,
                "passes": bool(pa_only["noninferior"]),
                "release_log_odds": release,
            }
            if cell["passes"]:
                deviation = _difference_sd(
                    [labels[offset] for offset in pa],
                    [native_scores[offset] for offset in pa],
                    [release[offset] for offset in pa],
                    [analysis_ids[offset] for offset in pa],
                )
                positives = sum(labels[offset] for offset in pa)
                se_reserved = se_at_n(deviation, 301, positives)
                cell["power"] = {
                    "bootstrap_sd": deviation,
                    "pa_positives": positives,
                    "power_at_0": power_noninferior(se_reserved, 0.0),
                    "power_at_minus_0_02": power_noninferior(se_reserved, -0.02),
                }
                cell["reserve_gate_open"] = cell["power"]["power_at_0"] >= 0.8
            else:
                cell["reserve_gate_open"] = False
            cells.append(cell)
            print(json.dumps({
                "prefix": size, "rank": name, "passes": cell["passes"],
                "pa_auc": pa_only["auc_release"], "lower": pa_only["one_sided_95_lower"],
            }), flush=True)
            del predicted
    chosen = [cell for cell in cells if cell["reserve_gate_open"]]
    result = {
        "schema": "nih_unlabeled_bridge_v1",
        "protocol_section": "2.27",
        "backend": BACKEND,
        "images_ready": 3000,
        "prefixes_scored": list(prefixes),
        "prefix_8000_scored": False,
        "attacker_scored": False,
        "primary_endpoint": "pa_only",
        "margin": 0.05,
        "pa_only_native_auc": pa_auc,
        "cells": [{key: value for key, value in cell.items() if key != "release_log_odds"} for cell in cells],
        "passing_cells": [
            {"prefix": cell["prefix"], "rank": cell["rank"]} for cell in cells if cell["passes"]
        ],
        "reserve_gate_open": bool(chosen),
        "confirmation_opened": False,
        "reserved_images_downloaded": False,
        "elapsed_s": time.perf_counter() - started,
        "limitations": [
            "The 8000 prefix is not scored. Its images beyond 3000 are not downloaded.",
            "The section 2.28 reconstructor trains on that 8000 prefix and is not fit.",
            "A passing cell does not download the reserved confirmation images in this file.",
        ],
    }
    DESTINATION.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "prefixes_scored": result["prefixes_scored"],
        "passing_cells": result["passing_cells"],
        "reserve_gate_open": result["reserve_gate_open"],
        "prefix_8000_scored": False,
    }), flush=True)


if __name__ == "__main__":
    main()
