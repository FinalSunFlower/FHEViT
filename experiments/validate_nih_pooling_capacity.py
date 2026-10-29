"""Is the pooled linear probe saturated, and does attention keep local signal?

EVALUATION_PROTOCOL.md sections 2.13 and 2.14. The PCA-4 target and the
patients stay those of section 2.10. Confirmation rows do not enter a penalty,
a component, or an epoch. No multilayer perceptron and no regional release.
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
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.diagnose_release_sufficiency import _fingerprint, per_study_metrics, summarize  # noqa: E402
from experiments.fit_nih_oracle_bridge import CONFIRMATION, DEVELOPMENT, _native, _records  # noqa: E402
from experiments.nih_pneumonia import (  # noqa: E402
    CONFIRMATION_NEGATIVE,
    CONFIRMATION_POSITIVE,
    REPLICATION_NEGATIVE,
    REPLICATION_POSITIVE,
)
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402
from experiments.validate_pretrained_anchor import ALPHAS, RELEASE_RANK, fit_prefix  # noqa: E402
from fpavlm import PrefixSpec  # noqa: E402
from fpavlm.evaluation import paired_metric_advantage  # noqa: E402
from fpavlm.nested_ridge import leave_one_patient_out, predict_ridge, select_ridge  # noqa: E402

BACKEND = "internvl2-2b"
PRIOR = ROOT / "experiments" / "results" / "nih_self_prediction_internvl2-2b.json"
STORED_PREFIX = 0.0337514943476229
KERNEL_ALPHAS = np.logspace(-2, 4, 7)
KERNEL_GAMMAS = np.logspace(-4, 1, 6)
POLY_RANK = 16
ATTENTION_EPOCHS = 100
ATTENTION_LR = 1e-3
ATTENTION_DECAY = 1e-2


def capacity_flags(kernel_beats, polynomial_beats, attention_beats):
    """The RBF row alone calls the linear probe insufficient. Attention is separate."""
    return {
        "linear_capacity_insufficient": bool(kernel_beats),
        "polynomial_reported_only": bool(polynomial_beats),
        "pooling_loss_supported": bool(attention_beats),
    }


def _standardize(train, rows):
    mean = train.mean(0)
    scale = train.std(0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (rows - mean) / scale, mean, scale


def select_kernel_ridge(features, targets, patients, alphas, gammas):
    """Leave-one-patient-out RBF ridge. Confirmation rows are not an argument."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if len(patients) != len(x):
        raise ValueError("patient ids must align with the training rows")
    folds = leave_one_patient_out(patients)
    best = None
    chosen = None
    for gamma in np.asarray(gammas, dtype=np.float64):
        for alpha in np.asarray(alphas, dtype=np.float64):
            errors = []
            for train_rows, held_rows in folds:
                fitted = _fit_kernel(x[train_rows], y[train_rows], float(alpha), float(gamma))
                residual = _predict_kernel(x[held_rows], fitted) - y[held_rows]
                errors.append(float((residual ** 2).sum(1).mean()))
            score = float(np.mean(errors))
            rank = (score, -float(alpha), float(gamma))
            if best is None or rank < best:
                best = rank
                chosen = {"alpha": float(alpha), "gamma": float(gamma), "cv_mse": score}
    chosen["model"] = _fit_kernel(x, y, chosen["alpha"], chosen["gamma"])
    finite_alpha = np.asarray(alphas, dtype=np.float64)
    finite_gamma = np.asarray(gammas, dtype=np.float64)
    chosen["alpha_at_boundary"] = bool(
        chosen["alpha"] in {float(finite_alpha.min()), float(finite_alpha.max())}
        or chosen["gamma"] in {float(finite_gamma.min()), float(finite_gamma.max())})
    return chosen


def _fit_kernel(train, targets, alpha, gamma):
    scaled, mean, scale = _standardize(train, train)
    model = KernelRidge(alpha=alpha, gamma=gamma, kernel="rbf")
    model.fit(scaled, targets)
    return {"model": model, "mean": mean, "scale": scale}


def _predict_kernel(rows, fitted):
    scaled = (np.asarray(rows, dtype=np.float64) - fitted["mean"]) / fitted["scale"]
    return fitted["model"].predict(scaled)


def polynomial_map(features, train_n, rank):
    """Degree-2 features of a development-only 16-component projection."""
    x = np.asarray(features, dtype=np.float64)
    scaled, _, _ = _standardize(x[:train_n], x)
    centered = scaled[:train_n] - scaled[:train_n].mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    if vt.shape[0] < rank:
        raise ValueError("the pooled features have fewer components than the locked rank")
    reduced = scaled @ vt[:rank].T
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(reduced[:train_n])
    return poly.transform(reduced)


class AttentionPool(torch.nn.Module):
    def __init__(self, hidden, rank):
        super().__init__()
        self.score = torch.nn.Linear(hidden, 1, bias=False)
        self.readout = torch.nn.Linear(hidden, rank)

    def forward(self, tokens):
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=-1)
        pooled = torch.einsum("nt,nth->nh", weights, tokens)
        return self.readout(pooled), weights


def train_attention(tokens, targets, train_n, readout_weight, readout_bias):
    torch.manual_seed(812)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = tokens.to(device)
    targets = targets.to(device)
    model = AttentionPool(tokens.shape[-1], targets.shape[-1]).to(device)
    with torch.no_grad():
        model.score.weight.zero_()
        model.readout.weight.copy_(torch.as_tensor(readout_weight, dtype=torch.float32, device=device))
        model.readout.bias.copy_(torch.as_tensor(readout_bias, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=ATTENTION_LR, weight_decay=ATTENTION_DECAY)
    train_tokens = tokens[:train_n]
    train_targets = targets[:train_n]
    model.train()
    for _ in range(ATTENTION_EPOCHS):
        optimizer.zero_grad()
        predicted, _ = model(train_tokens)
        torch.nn.functional.mse_loss(predicted, train_targets).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        predicted, weights = model(tokens)
    return predicted.detach().cpu(), weights.detach().cpu()


def _prefix_gain(codes, native, train, test, patients, train_mean, spec):
    predicted = fit_prefix(codes, native, train, test, spec)
    summary = summarize("prefix", predicted, native[test], train_mean, patients, bootstrap_samples=2000)
    gain = per_study_metrics(predicted, native[test], train_mean)["mse_improvement"].numpy()
    return summary, gain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=[BACKEND], default=BACKEND)
    args = parser.parse_args()
    if args.backend != BACKEND:
        raise SystemExit("sections 2.13 and 2.14 fit internvl2-2b only")
    if not PRIOR.is_file():
        raise SystemExit("section 2.10 result is not on disk")
    torch.set_num_threads(4)
    torch.manual_seed(812)
    started = time.perf_counter()
    development = _records(DEVELOPMENT, REPLICATION_POSITIVE + REPLICATION_NEGATIVE)
    confirmation = _records(CONFIRMATION, CONFIRMATION_POSITIVE + CONFIRMATION_NEGATIVE)
    dev_ids = [row["patient_id"] for row in development]
    conf_ids = [row["patient_id"] for row in confirmation]
    if set(dev_ids) & set(conf_ids):
        raise ValueError("development and confirmation patients overlap")
    llm = load_backend(BACKEND)
    native = torch.cat([
        _native(llm, DEVELOPMENT, development, BACKEND),
        _native(llm, CONFIRMATION, confirmation, BACKEND),
    ]).float()
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_n = len(development)
    train = list(range(train_n))
    test = list(range(train_n, len(native)))
    flat = native.reshape(len(native), -1).double()
    centered = flat - flat[:train_n].mean(0)
    basis = torch.linalg.svd(centered[:train_n], full_matrices=False).Vh.T[:, :RELEASE_RANK]
    oracle = (centered @ basis).float()
    pooled = native.mean(dim=1).numpy()
    tokens, hidden = int(native.shape[-2]), int(native.shape[-1])
    digest = hashlib.sha256("\n".join(dev_ids).encode("utf-8")).hexdigest()
    token_mean = native[:train_n].mean(0, keepdim=True)

    def spec_for(name):
        return PrefixSpec(
            BACKEND, (RELEASE_RANK, 1), tokens, hidden, digest,
            _fingerprint({"release": name, "rank": RELEASE_RANK, "section": "2.13"}),
        )

    linear = select_ridge(pooled[:train_n], oracle[:train_n].numpy(), dev_ids, ALPHAS)
    linear_codes = torch.as_tensor(predict_ridge(pooled, linear), dtype=torch.float32)
    linear_summary, linear_gain = _prefix_gain(
        linear_codes, native, train, test, conf_ids, token_mean, spec_for("linear"))
    if abs(linear_summary["mse_improvement_over_mean"] - STORED_PREFIX) > 0.001:
        raise SystemExit("the refit linear prefix disagrees with section 2.10")
    coordinate_mean = oracle[:train_n].mean(0, keepdim=True)

    def pack(name, codes, model_meta):
        summary, gain = _prefix_gain(codes, native, train, test, conf_ids, token_mean, spec_for(name))
        coordinate = summarize(
            name, codes[test], oracle[test], coordinate_mean, conf_ids, bootstrap_samples=2000)
        paired = paired_metric_advantage(gain, linear_gain, conf_ids)
        return {
            **model_meta,
            "coordinate": coordinate,
            "prefix": summary,
            "versus_linear": paired,
            "beats_linear_prefix": bool(paired["significant_positive"]),
        }

    kernel = select_kernel_ridge(
        pooled[:train_n], oracle[:train_n].numpy(), dev_ids, KERNEL_ALPHAS, KERNEL_GAMMAS)
    kernel_codes = torch.as_tensor(_predict_kernel(pooled, kernel["model"]), dtype=torch.float32)
    kernel_row = pack("rbf", kernel_codes, {
        "alpha": kernel["alpha"], "gamma": kernel["gamma"],
        "alpha_at_boundary": kernel["alpha_at_boundary"],
    })
    poly = polynomial_map(pooled, train_n, POLY_RANK)
    poly_model = select_ridge(poly[:train_n], oracle[:train_n].numpy(), dev_ids, ALPHAS)
    poly_codes = torch.as_tensor(predict_ridge(poly, poly_model), dtype=torch.float32)
    poly_row = pack("polynomial", poly_codes, {
        "components": POLY_RANK,
        "alpha": None if poly_model["intercept_only"] else poly_model["alpha"],
        "intercept_only": poly_model["intercept_only"],
        "alpha_at_boundary": poly_model["alpha_at_boundary"],
    })
    attention_codes, attention_weights = train_attention(
        native, oracle, train_n, linear["weight"].T, linear["bias"])
    attention_row = pack("attention", attention_codes, {
        "epochs": ATTENTION_EPOCHS,
        "learning_rate": ATTENTION_LR,
        "weight_decay": ATTENTION_DECAY,
        "confirmation_mean_max_weight": float(attention_weights[test].max(dim=1).values.mean()),
    })
    flags = capacity_flags(
        kernel_row["beats_linear_prefix"],
        poly_row["beats_linear_prefix"],
        attention_row["beats_linear_prefix"],
    )
    result = {
        "schema": "nih_pooling_capacity_v1",
        "protocol_sections": ["2.13", "2.14"],
        "backend": BACKEND,
        "development_patients": train_n,
        "confirmation_patients": len(confirmation),
        "fit_saw_confirmation": False,
        "linear_prefix": linear_summary["mse_improvement_over_mean"],
        "linear_alpha": None if linear["intercept_only"] else linear["alpha"],
        "rbf": {key: value for key, value in kernel_row.items()},
        "polynomial": poly_row,
        "attention": attention_row,
        **flags,
        "mlp_started": False,
        "regional_release_started": False,
        "qwen_fitted": False,
        "official_test_list_read": False,
        "limitations": [
            "The polynomial row first keeps 16 development components, so it is not the capacity call.",
            "Attention is trained for a locked epoch count. Confirmation loss did not pick the epoch.",
            "A multilayer perceptron and a regional release are not started from this result.",
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    destination = ROOT / "experiments" / "results" / "nih_pooling_capacity_internvl2-2b.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "linear_prefix": result["linear_prefix"],
        "rbf_prefix": kernel_row["prefix"]["mse_improvement_over_mean"],
        "rbf_beats_linear": kernel_row["beats_linear_prefix"],
        "polynomial_prefix": poly_row["prefix"]["mse_improvement_over_mean"],
        "polynomial_beats_linear": poly_row["beats_linear_prefix"],
        "attention_prefix": attention_row["prefix"]["mse_improvement_over_mean"],
        "attention_beats_linear": attention_row["beats_linear_prefix"],
        "linear_capacity_insufficient": flags["linear_capacity_insufficient"],
        "pooling_loss_supported": flags["pooling_loss_supported"],
    }, indent=2))


if __name__ == "__main__":
    main()
