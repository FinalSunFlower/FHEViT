"""Plaintext polynomial InternViT. Not a circuit candidate.

EVALUATION_PROTOCOL.md section 2.39. Nonlinearities are replaced before any
score. The reserved confirmation set is not used.
"""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.nih_redraw_reading import REDRAWS  # noqa: E402
from experiments.validate_nih_fine_noise import DESTINATION as FINE_NOISE  # noqa: E402
from experiments.validate_nih_oracle_auc import _log_odds  # noqa: E402
from experiments.validate_nih_rank1999_redraws import _bridge, _gram, _project, training_draws  # noqa: E402
from experiments.validate_nih_unlabeled_bridge import (  # noqa: E402
    POOL,
    SEALED,
    WIDTH,
    _auc_block,
    pca_from_gram,
)
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402
from experiments.validate_pneumonia_log_odds import NEGATIVE, POSITIVE, single_token_id  # noqa: E402
from fpavlm.evaluation import roc_auc  # noqa: E402

BACKEND = "internvl2-2b"
RANKS = (1024, 1999)
BOOTSTRAPS = 73
MULTIPLICATIVE_DEPTH = 172
HOLDOUT = ROOT / "data" / "nih_chestxray14_final_holdout" / "positives.json"
STORED = ROOT / "experiments" / "results" / "nih_sealed_confirmation_auc_internvl2-2b.json"
STRATA = ROOT / "experiments" / "results" / "nih_sealed_strata.json"
DESTINATION = ROOT / "experiments" / "results" / "nih_poly_tower_internvl2-2b.json"
TRAIN_CACHE = POOL / "poly_token_cache.float32"
ANALYSIS_CACHE = POOL / "analysis_poly_token_cache.float32"


def gelu_polynomial(values):
    return 0.1365 + 0.5 * values + 0.2045 * values * values


def rsqrt_polynomial(variance):
    shift = variance - 1
    return 1 - 0.5 * shift + 0.375 * shift * shift


def _poly_layernorm(module, hidden):
    work = hidden.float()
    mean = work.mean(dim=-1, keepdim=True)
    variance = work.var(dim=-1, unbiased=False, keepdim=True)
    centered = (work - mean) * rsqrt_polynomial(variance)
    weight = module.weight.float()
    bias = 0 if module.bias is None else module.bias.float()
    return (centered * weight + bias).to(hidden.dtype)


def _poly_gelu(module, hidden):
    del module
    return gelu_polynomial(hidden.float()).to(hidden.dtype)


def _poly_attention(self, hidden):
    batch, tokens, channels = hidden.shape
    qkv = self.qkv(hidden).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
    query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    scores = (query * self.scale) @ key.transpose(-2, -1)
    weights = scores * scores
    denom = weights.sum(dim=-1, keepdim=True)
    if torch.any(denom == 0):
        raise RuntimeError("a polynomial attention row summed to zero")
    mixed = (weights / denom) @ value
    mixed = mixed.transpose(1, 2).reshape(batch, tokens, channels)
    return self.proj_drop(self.proj(mixed))


def patch_tower(model):
    """Replace nonlinearities on the vision tower and the release projector only."""
    norms = gels = attentions = 0
    for root in (model.vision_model, model.mlp1):
        for module in root.modules():
            name = module.__class__.__name__
            if name == "LayerNorm":
                module.forward = types.MethodType(_poly_layernorm, module)
                norms += 1
            elif name in {"GELU", "GELUActivation"}:
                module.forward = types.MethodType(_poly_gelu, module)
                gels += 1
            elif name == "InternAttention":
                module.use_flash_attn = False
                module._naive_attn = types.MethodType(_poly_attention, module)
                attentions += 1
    if (norms, gels, attentions) != (49, 25, 24):
        raise SystemExit(f"polynomial replacements were {norms}, {gels}, {attentions}, not 49, 25, 24")
    return {"layernorms": norms, "gelus": gels, "attentions": attentions}


def _pixels(path):
    from PIL import Image

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((448, 448), Image.Resampling.BILINEAR), dtype=np.float32)
    array = (array / 255.0 - mean) / std
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)


def _divergence(model, image_path):
    """First encoder block whose output is not finite. Coefficients stay frozen."""
    vision = model.vision_model
    pixels = _pixels(image_path).to("cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        hidden = vision.embeddings(pixels)
        for index, layer in enumerate(vision.encoder.layers):
            hidden = layer(hidden)
            if not bool(torch.isfinite(hidden).all()):
                return {"finite": False, "block": index}
    return {"finite": True, "block": None}


def _fill(model, root, records, path):
    if path.is_file() and path.stat().st_size == len(records) * WIDTH * 4:
        cached = np.memmap(path, dtype=np.float32, mode="r", shape=(len(records), WIDTH))
        if not np.isfinite(np.asarray(cached[:1])).all():
            raise SystemExit("the polynomial cache is not finite and is not reused")
        return cached
    cache = np.memmap(path, dtype=np.float32, mode="w+", shape=(len(records), WIDTH))
    for index, row in enumerate(records):
        pixels = _pixels(root / row["relative_path"]).to("cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            tokens = model.extract_feature(pixels)
        if tuple(tokens.shape) != (1, 256, 2048):
            raise SystemExit(f"polynomial tokens have shape {tuple(tokens.shape)}")
        flat = tokens.float().cpu().numpy().reshape(-1)
        if not np.isfinite(flat).all():
            raise SystemExit("polynomial tokens are not finite")
        cache[index] = flat
        if index % 50 == 0:
            print(f"polynomial {path.name} {index}/{len(records)}", flush=True)
    cache.flush()
    return cache


def _write(payload):
    DESTINATION.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    torch.set_num_threads(4)
    started = time.perf_counter()
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    if holdout.get("images_downloaded") or holdout.get("auc_computed") or holdout.get("negative_sample_drawn"):
        raise SystemExit("the reserved confirmation pool was already opened")
    pool = json.loads((POOL / "metadata.json").read_text(encoding="utf-8"))
    redraws = json.loads(REDRAWS.read_text(encoding="utf-8"))
    train_ids = [str(row["patient_id"]) for row in pool["records"][:3000]]
    draws = training_draws(3000, 2000, 5, 42)
    for draw, cell in zip(draws, redraws["cells"]):
        if [train_ids[int(index)] for index in draw] != cell["patient_ids"]:
            raise SystemExit("a polynomial draw does not match the stored five")
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
    print("loading the backend", flush=True)
    llm = load_backend(BACKEND)
    replaced = patch_tower(llm.model)
    trace = _divergence(llm.model, POOL / pool["records"][0]["relative_path"])
    if not trace["finite"]:
        payload = {
            "schema": "nih_poly_tower_v1",
            "protocol_section": "2.39",
            "circuit_candidate": False,
            "bootstraps": BOOTSTRAPS,
            "multiplicative_depth": MULTIPLICATIVE_DEPTH,
            "replacements": replaced,
            "gelu": [0.1365, 0.5, 0.2045],
            "tokens_finite": False,
            "first_nonfinite_block": trace["block"],
            "auc_computed": False,
            "utility_holds": False,
            "confirmation_opened": False,
            "reserved_images_downloaded": False,
            "complete": True,
            "elapsed_s": round(time.perf_counter() - started, 1),
        }
        _write(payload)
        print(json.dumps({"tokens_finite": False, "first_nonfinite_block": trace["block"]}), flush=True)
        return
    positive_id = single_token_id(llm.tokenizer, POSITIVE)
    negative_id = single_token_id(llm.tokenizer, NEGATIVE)
    train_cache = _fill(llm.model, POOL, pool["records"][:3000], TRAIN_CACHE)
    analysis_cache = _fill(llm.model, SEALED, analysis, ANALYSIS_CACHE)
    payload = {
        "schema": "nih_poly_tower_v1",
        "protocol_section": "2.39",
        "circuit_candidate": False,
        "bootstraps": BOOTSTRAPS,
        "multiplicative_depth": MULTIPLICATIVE_DEPTH,
        "replacements": replaced,
        "gelu": [0.1365, 0.5, 0.2045],
        "confirmation_opened": False,
        "reserved_images_downloaded": False,
        "cells": [],
        "complete": False,
    }
    done = set()
    if DESTINATION.is_file():
        existing = json.loads(DESTINATION.read_text(encoding="utf-8"))
        if existing.get("complete") is True and existing.get("bootstraps") == BOOTSTRAPS:
            print("polynomial tower already complete", flush=True)
            return
        payload["cells"] = existing.get("cells", [])
        done = {(item["draw"], item["rank"]) for item in payload["cells"]}
    for number, rows in enumerate(draws):
        mean, gram = _gram(train_cache, rows)
        full = pca_from_gram(gram, 1999)
        for rank in RANKS:
            if (number, rank) in done:
                continue
            train_code, test_code = _project(train_cache, rows, analysis_cache, mean, gram, full[:, :rank])
            predicted = _bridge(train_code, train_cache, rows, test_code)
            release = [
                _log_odds(llm, predicted[offset].unsqueeze(0), positive_id, negative_id)
                for offset in range(predicted.shape[0])
            ]
            block = _auc_block(
                [labels[offset] for offset in pa],
                [native_scores[offset] for offset in pa],
                [release[offset] for offset in pa],
                [analysis_ids[offset] for offset in pa],
            )
            cell = {
                "draw": number,
                "rank": rank,
                "pa_only": block,
                "passes": bool(block["noninferior"]),
            }
            payload["cells"].append(cell)
            payload["elapsed_s"] = round(time.perf_counter() - started, 1)
            _write(payload)
            print(json.dumps({
                "draw": number,
                "rank": rank,
                "difference": block["difference"],
                "lower": block["one_sided_95_lower"],
                "passes": cell["passes"],
            }), flush=True)
            del predicted
    summaries = []
    for rank in RANKS:
        passes = [item["passes"] for item in payload["cells"] if item["rank"] == rank]
        if len(passes) != 5:
            raise SystemExit(f"rank {rank} does not have five draws")
        summaries.append({
            "rank": rank,
            "refit_passes": int(sum(passes)),
            "draws": 5,
            "passes": sum(passes) >= 4,
        })
    payload["summaries"] = summaries
    payload["utility_holds"] = any(item["passes"] for item in summaries)
    if FINE_NOISE.is_file():
        budget = json.loads(FINE_NOISE.read_text(encoding="utf-8"))
        payload["error_budget"] = budget.get("error_budget") if budget.get("complete") else None
    else:
        payload["error_budget"] = None
    payload["complete"] = True
    payload["circuit_candidate"] = False
    payload["confirmation_opened"] = False
    payload["reserved_images_downloaded"] = False
    payload["elapsed_s"] = round(time.perf_counter() - started, 1)
    _write(payload)
    print(json.dumps({
        "complete": True,
        "utility_holds": payload["utility_holds"],
        "bootstraps": BOOTSTRAPS,
        "multiplicative_depth": MULTIPLICATIVE_DEPTH,
    }), flush=True)


if __name__ == "__main__":
    main()
