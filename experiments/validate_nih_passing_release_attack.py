"""Reconstruction and attribute attack on the two passing releases.

EVALUATION_PROTOCOL.md section 2.36. Train on the 3000 images. Score the
506 analysis patients. The reserved confirmation set is not used.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.validate_nih_attribute_inference import (  # noqa: E402
    _age_interval,
    _binary_interval,
    _fit_age,
    _fit_binary,
    _targets,
    _valid_draws,
)
from experiments.validate_nih_unlabeled_bridge import (  # noqa: E402
    ANALYSIS_CACHE,
    POOL,
    SEALED,
    gram_from_rows,
    pca_from_gram,
    project_rows,
)

BACKEND = "internvl2-2b"
TRAIN_COUNT = 3000
BASIS_COUNT = 2000
ANALYSIS_COUNT = 506
RANK = 1999
NARROW_RANK = 1024
EPOCHS = 20
BATCH = 32
DRAWS = 2000
BOOTSTRAP_SEED = 42
MODEL_SEED = 812
WIDTH = 256 * 2048
HOLDOUT = ROOT / "data" / "nih_chestxray14_final_holdout" / "positives.json"
OUTPUT = ROOT / "experiments" / "results" / f"nih_passing_release_attack_{BACKEND}.json"


class Decoder(torch.nn.Module):
    """Section 2.28 decoder. One ReLU, then two transposed convolutions."""

    def __init__(self, kind, in_features):
        super().__init__()
        if kind not in {"tokens", "coordinates"}:
            raise ValueError(f"unknown attacker input: {kind}")
        self.kind = kind
        if kind == "tokens":
            self.stem = torch.nn.Conv2d(in_features, 128, kernel_size=1)
            self.vector = None
        else:
            self.stem = None
            self.vector = torch.nn.Linear(in_features, 16 * 16 * 128)
        self.up1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up2 = torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, features):
        if self.kind == "tokens":
            batch, _tokens, channels = features.shape
            grid = features.transpose(1, 2).reshape(batch, channels, 16, 16)
            hidden = self.stem(grid)
        else:
            hidden = self.vector(features).reshape(features.shape[0], 128, 16, 16)
        hidden = torch.relu(hidden)
        hidden = self.up1(hidden)
        return torch.sigmoid(self.up2(hidden))


def _images(root, records):
    images = np.empty((len(records), 1, 64, 64), dtype=np.float32)
    for index, row in enumerate(records):
        with Image.open(root / row["relative_path"]) as image:
            gray = image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
            images[index, 0] = np.asarray(gray, dtype=np.float32) / 255.0
    return images


def _token_mean(cache, count):
    pieces = []
    for start in range(0, count, 64):
        block = np.asarray(cache[start:start + 64], dtype=np.float32).reshape(-1, 256, 2048)
        pieces.append(block.mean(axis=1))
    return np.concatenate(pieces, axis=0)


def _interval(values):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.asarray(values, dtype=float)
    samples = np.empty(DRAWS, dtype=float)
    for index in range(DRAWS):
        chosen = rng.choice(len(values), size=len(values), replace=True)
        samples[index] = values[chosen].mean()
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "beats_baseline": bool(lower > 0.0),
    }


def _ssim_psnr(prediction, target):
    forecast = np.asarray(prediction, dtype=np.float64)[0]
    gold = np.asarray(target, dtype=np.float64)[0]
    mse = float(np.mean((forecast - gold) ** 2))
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))
    ssim = float(structural_similarity(gold, forecast, data_range=1.0))
    return ssim, psnr


def _predict(model, kind, features):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(features), BATCH):
            if kind == "tokens":
                batch = np.asarray(features[start:start + BATCH], dtype=np.float32).reshape(-1, 256, 2048)
                source = torch.from_numpy(batch)
            else:
                source = torch.from_numpy(np.asarray(features[start:start + BATCH], dtype=np.float32))
            pieces.append(model(source.cuda()).cpu().numpy())
    return np.concatenate(pieces, axis=0)


def _train(kind, features, images):
    if not torch.cuda.is_available():
        raise RuntimeError("the section 2.36 attacker requires the GPU")
    torch.manual_seed(MODEL_SEED)
    in_features = 2048 if kind == "tokens" else int(features.shape[1])
    model = Decoder(kind, in_features).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    generator = torch.Generator()
    generator.manual_seed(MODEL_SEED)
    images_t = torch.from_numpy(images)
    coordinates = None if kind == "tokens" else torch.from_numpy(np.asarray(features, dtype=np.float32))
    count = len(images)
    model.train()
    for epoch in range(EPOCHS):
        order = torch.randperm(count, generator=generator)
        total = 0.0
        for start in range(0, count, BATCH):
            chosen = order[start:start + BATCH]
            target = images_t[chosen].cuda()
            if kind == "tokens":
                batch = np.asarray(features[chosen.numpy()], dtype=np.float32).reshape(-1, 256, 2048)
                source = torch.from_numpy(batch).cuda()
            else:
                source = coordinates[chosen].cuda()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(source), target)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * int(chosen.shape[0])
        print(f"{kind} epoch {epoch + 1}/{EPOCHS} mse {total / count:.6f}", flush=True)
    return model


def _score_release(kind, model, test_features, train_images, test_images, train_views, test_views):
    prediction = _predict(model, kind, test_features)
    mean_image = train_images.mean(axis=0)
    view_means = {}
    for view in ("PA", "AP"):
        mask = np.asarray(train_views) == view
        if not bool(mask.any()):
            raise ValueError(f"training images have no {view} view")
        view_means[view] = train_images[mask].mean(axis=0)
    recon_ssim = []
    recon_psnr = []
    mean_ssim = []
    view_ssim = []
    for index in range(len(test_images)):
        ssim, psnr = _ssim_psnr(prediction[index], test_images[index])
        recon_ssim.append(ssim)
        recon_psnr.append(psnr)
        mean_ssim.append(_ssim_psnr(mean_image, test_images[index])[0])
        view_ssim.append(_ssim_psnr(view_means[test_views[index]], test_images[index])[0])
    recon_ssim = np.asarray(recon_ssim)
    mean_ssim = np.asarray(mean_ssim)
    view_ssim = np.asarray(view_ssim)
    return {
        "input": kind,
        "ssim_mean": float(recon_ssim.mean()),
        "psnr_mean": float(np.mean(recon_psnr)),
        "versus_training_mean": _interval(recon_ssim - mean_ssim),
        "versus_view_conditional_mean": _interval(recon_ssim - view_ssim),
        "primary_baseline": "view_conditional_mean",
    }


def _attribute_block(name, train_x, test_x, train_targets, test_targets, draw_bank, age_draws):
    sex = _binary_interval(
        test_targets["sex_female"],
        _fit_binary(train_x, train_targets["sex_female"], test_x),
        draw_bank["sex_female"],
    )
    view = _binary_interval(
        test_targets["view_pa"],
        _fit_binary(train_x, train_targets["view_pa"], test_x),
        draw_bank["view_pa"],
    )
    age_pred, age_baseline = _fit_age(train_x, train_targets["age"], test_x)
    age = _age_interval(test_targets["age"], age_pred, age_baseline, age_draws)
    for block in (sex, view):
        block.pop("samples", None)
    return {
        "features": name,
        "sex_female": {key: value for key, value in sex.items() if key != "samples"},
        "view_pa": {key: value for key, value in view.items() if key != "samples"},
        "age": age,
        "training_mean_age": age_baseline,
    }


def _coordinates(train_cache, analysis_cache):
    basis = train_cache[:BASIS_COUNT]
    mean, gram = gram_from_rows(basis)
    coeff = pca_from_gram(gram, RANK)
    head, tail = project_rows(basis, train_cache[BASIS_COUNT:TRAIN_COUNT], mean, gram, coeff)
    _ignored, analysis = project_rows(basis, analysis_cache, mean, gram, coeff)
    del _ignored
    return np.concatenate([head, tail], axis=0).astype(np.float32), analysis.astype(np.float32)


def _write(payload):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    started = time.time()
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    if holdout.get("images_downloaded") or holdout.get("auc_computed") or holdout.get("negative_sample_drawn"):
        raise RuntimeError("reserved confirmation was opened")
    payload = {
        "schema": "nih_passing_release_attack_v1",
        "protocol_section": "2.36",
        "backend": BACKEND,
        "train_count": TRAIN_COUNT,
        "basis_count": BASIS_COUNT,
        "analysis_count": ANALYSIS_COUNT,
        "ranks": [NARROW_RANK, RANK],
        "noise_scale": 0,
        "epochs": EPOCHS,
        "batch_size": BATCH,
        "seed": MODEL_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confirmation_opened": False,
        "reserved_images_downloaded": False,
        "chooses_release": False,
        "complete": False,
        "releases": {},
        "attributes": {},
    }
    if OUTPUT.is_file():
        existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if existing.get("complete") is True:
            print("attack already complete", flush=True)
            return
        payload["releases"] = existing.get("releases", {})
        payload["attributes"] = existing.get("attributes", {})

    pool = json.loads((POOL / "metadata.json").read_text(encoding="utf-8"))
    analysis_meta = json.loads((SEALED / "metadata.json").read_text(encoding="utf-8"))
    train_records = pool["records"][:TRAIN_COUNT]
    test_records = analysis_meta["records"]
    if len(test_records) != ANALYSIS_COUNT:
        raise ValueError(f"expected {ANALYSIS_COUNT} analysis patients, found {len(test_records)}")
    train_cache = np.memmap(POOL / "token_cache.float32", dtype=np.float32, mode="r", shape=(TRAIN_COUNT, WIDTH))
    analysis_cache = np.memmap(
        ANALYSIS_CACHE, dtype=np.float32, mode="r", shape=(ANALYSIS_COUNT, WIDTH),
    )
    print("fitting the rank-1999 basis on the first 2000", flush=True)
    train_codes, test_codes = _coordinates(train_cache, analysis_cache)
    print("loading images", flush=True)
    train_images = _images(POOL, train_records)
    test_images = _images(SEALED, test_records)
    train_views = [row["view_position"] for row in train_records]
    test_views = [row["view_position"] for row in test_records]

    if not payload["attributes"]:
        print("fitting attribute models", flush=True)
        train_targets = _targets(train_records)
        test_targets = _targets(test_records)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draw_bank = {
            "sex_female": _valid_draws(test_targets["sex_female"], ANALYSIS_COUNT, rng),
            "view_pa": _valid_draws(test_targets["view_pa"], ANALYSIS_COUNT, rng),
        }
        age_draws = [rng.choice(ANALYSIS_COUNT, size=ANALYSIS_COUNT, replace=True) for _ in range(DRAWS)]
        feature_sets = {
            "token_mean": (_token_mean(train_cache, TRAIN_COUNT), _token_mean(analysis_cache, ANALYSIS_COUNT)),
            "rank_1024": (train_codes[:, :NARROW_RANK], test_codes[:, :NARROW_RANK]),
            "rank_1999": (train_codes, test_codes),
        }
        for name, (train_x, test_x) in feature_sets.items():
            block = _attribute_block(
                name, train_x, test_x, train_targets, test_targets, draw_bank, age_draws,
            )
            payload["attributes"][name] = block
            print(
                f"attribute {name} sex {block['sex_female']['auc']:.3f} "
                f"view {block['view_pa']['auc']:.3f} age mae {block['age']['mae']:.3f}",
                flush=True,
            )
        _write(payload)

    jobs = (
        ("tokens", "tokens", train_cache, analysis_cache),
        ("rank_1024", "coordinates", train_codes[:, :NARROW_RANK], test_codes[:, :NARROW_RANK]),
        ("rank_1999", "coordinates", train_codes, test_codes),
    )
    for name, kind, train_features, test_features in jobs:
        if name in payload["releases"]:
            print(f"skipping finished release {name}", flush=True)
            continue
        print(f"training {name}", flush=True)
        model = _train(kind, train_features, train_images)
        payload["releases"][name] = _score_release(
            kind, model, test_features, train_images, test_images, train_views, test_views,
        )
        primary = payload["releases"][name]["versus_view_conditional_mean"]
        print(
            f"{name} ssim {payload['releases'][name]['ssim_mean']:.4f} "
            f"view advantage lower {primary['ci95_lower']:.4f} beats {primary['beats_baseline']}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()
        _write(payload)

    payload["complete"] = True
    payload["elapsed_s"] = round(time.time() - started, 1)
    _write(payload)
    print(json.dumps({"complete": True, "elapsed_s": payload["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    main()
