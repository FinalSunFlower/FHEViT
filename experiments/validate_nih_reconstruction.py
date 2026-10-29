"""Linear reconstruction from tokens, their mean, and low-rank coordinates.

EVALUATION_PROTOCOL.md section 2.26. Fit on the 72 development patients.
Score the 506 analysis patients. The reserved confirmation set is not used.
"""
from __future__ import annotations

import argparse
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

from experiments.fit_nih_oracle_bridge import DEVELOPMENT, _native, _records  # noqa: E402
from experiments.nih_pneumonia import REPLICATION_NEGATIVE, REPLICATION_POSITIVE  # noqa: E402
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402

BACKEND = "internvl2-2b"
RANKS = (4, 8, 16)
DRAWS = 2000
SEED = 42
SEALED = ROOT / "data" / "nih_chestxray14_sealed_confirmation"
HOLDOUT = ROOT / "data" / "nih_chestxray14_final_holdout" / "positives.json"


def reconstruct(train_x, train_y, test_x, alpha=1.0):
    """Ridge with training-set standardization. The weight matrix is not stored."""
    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    centered = (train_x - mean) / scale
    target_mean = train_y.mean(axis=0)
    gram = centered @ centered.T + alpha * np.eye(len(centered))
    dual = np.linalg.solve(gram, train_y - target_mean)
    pieces = []
    test_x = np.asarray(test_x, dtype=np.float64)
    for start in range(0, len(test_x), 32):
        block = (test_x[start:start + 32] - mean) / scale
        pieces.append(block @ centered.T @ dual + target_mean)
    return np.clip(np.concatenate(pieces, axis=0), 0.0, 1.0)


def _image(path):
    with Image.open(path) as image:
        gray = image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
        return np.asarray(gray, dtype=np.float64) / 255.0


def _psnr(prediction, target):
    mse = float(np.mean((prediction - target) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _ssim(prediction, target):
    return float(structural_similarity(target, prediction, data_range=1.0))


def _interval(values):
    samples = []
    rng = np.random.default_rng(SEED)
    values = np.asarray(values, dtype=float)
    for _ in range(DRAWS):
        chosen = rng.choice(len(values), size=len(values), replace=True)
        samples.append(float(values[chosen].mean()))
    samples = np.asarray(samples, dtype=float)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci95_lower": float(lower), "ci95_upper": float(upper)}


def _lpips_values(loss, prediction, target):
    pred = torch.from_numpy(np.repeat(prediction[:, None], 3, axis=1)).float()
    true = torch.from_numpy(np.repeat(target[:, None], 3, axis=1)).float()
    pred = pred * 2 - 1
    true = true * 2 - 1
    scores = []
    for start in range(0, len(pred), 32):
        batch_pred = pred[start:start + 32]
        batch_true = true[start:start + 32]
        scores.append(loss(batch_pred, batch_true).detach().cpu().flatten())
    return torch.cat(scores).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=[BACKEND], default=BACKEND)
    args = parser.parse_args()
    if args.backend != BACKEND:
        raise SystemExit("section 2.26 fits internvl2-2b only")
    import lpips

    torch.set_num_threads(4)
    torch.manual_seed(812)
    started = time.perf_counter()
    development = _records(DEVELOPMENT, REPLICATION_POSITIVE + REPLICATION_NEGATIVE)
    confirmation = json.loads((SEALED / "metadata.json").read_text(encoding="utf-8"))["records"]
    holdout = {row["patient_id"] for row in json.loads(HOLDOUT.read_text(encoding="utf-8"))["records"]}
    if set(row["patient_id"] for row in development) & set(row["patient_id"] for row in confirmation):
        raise SystemExit("development and the analysis set overlap")
    if holdout & {row["patient_id"] for row in development + confirmation}:
        raise SystemExit("a reserved confirmation patient entered the attack")
    if len(development) != 72 or len(confirmation) != 506:
        raise SystemExit("the attack split is not 72 and 506")
    targets = np.stack([
        *(_image(DEVELOPMENT / row["relative_path"]) for row in development),
        *(_image(SEALED / row["relative_path"]) for row in confirmation),
    ])
    llm = load_backend(BACKEND)
    native = torch.cat([
        _native(llm, DEVELOPMENT, development, BACKEND),
        _native(llm, SEALED, confirmation, BACKEND),
    ]).float()
    train_n = len(development)
    flat = native.reshape(len(native), -1).double()
    centered = flat - flat[:train_n].mean(0)
    basis = torch.linalg.svd(centered[:train_n], full_matrices=False).Vh.T
    features = {
        "tokens": flat.float().numpy(),
        "pooled": native.mean(1).numpy(),
    }
    for rank in RANKS:
        features[f"pca_rank_{rank}"] = (centered @ basis[:, :rank]).float().numpy()
    train_y = targets[:train_n].reshape(train_n, -1)
    test_y = targets[train_n:]
    loss = lpips.LPIPS(net="alex")
    rows = []
    for name, matrix in features.items():
        predicted = reconstruct(matrix[:train_n], train_y, matrix[train_n:]).reshape(len(confirmation), 64, 64)
        psnr = [_psnr(predicted[index], test_y[index]) for index in range(len(confirmation))]
        ssim = [_ssim(predicted[index], test_y[index]) for index in range(len(confirmation))]
        perceptual = _lpips_values(loss, predicted, test_y)
        rows.append({
            "input": name,
            "psnr": _interval(psnr),
            "ssim": _interval(ssim),
            "lpips": _interval(perceptual),
        })
        print(json.dumps({
            "input": name,
            "psnr": rows[-1]["psnr"]["mean"],
            "ssim": rows[-1]["ssim"]["mean"],
            "lpips": rows[-1]["lpips"]["mean"],
        }), flush=True)
    result = {
        "schema": "nih_reconstruction_ceiling_v1",
        "protocol_section": "2.26",
        "backend": BACKEND,
        "development_patients": train_n,
        "analysis_patients": len(confirmation),
        "reserved_confirmation_used": False,
        "image_size": [64, 64],
        "attacks": rows,
        "release_chosen": False,
        "grid_scored": False,
        "finite_state_symbol_tested": False,
        "predictor_retarget_started": False,
        "elapsed_s": time.perf_counter() - started,
        "limitations": [
            "The attacker is linear. A nonlinear reconstructor is not trained.",
            "These patients are the scored analysis set, not the reserved confirmation set.",
        ],
    }
    destination = ROOT / "experiments" / "results" / "nih_reconstruction_internvl2-2b.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
