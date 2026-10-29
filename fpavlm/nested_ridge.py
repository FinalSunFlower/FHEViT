"""Patient-level ridge with the penalty chosen inside the training patients.

Standardized inputs make the penalty comparable across feature scales. Leave-one-
patient-out scores every penalty, and a tie keeps the stronger one. Held-out
patients are not an argument of the selector.
"""
from __future__ import annotations

import numpy as np


def leave_one_patient_out(patients):
    groups = np.asarray(patients)
    if groups.ndim != 1 or len(groups) < 2:
        raise ValueError("patient ids must align with at least two rows")
    names = list(dict.fromkeys(groups.tolist()))
    if len(names) < 2:
        raise ValueError("at least two patients are required")
    folds = []
    for name in names:
        held = groups == name
        folds.append((np.flatnonzero(~held), np.flatnonzero(held)))
    return tuple(folds)


def _as_matrices(features, targets):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("aligned feature and target matrices are required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and targets must be finite")
    return x, y


def fit_ridge(features, targets, alpha: float):
    if not np.isfinite(alpha) or alpha <= 0:
        if not np.isinf(alpha):
            raise ValueError("the ridge penalty must be a positive finite number or infinity")
    x, y = _as_matrices(features, targets)
    mean, y_mean = x.mean(0), y.mean(0)
    if np.isinf(alpha):
        return {"mean": mean, "scale": np.ones(x.shape[1]), "weight": np.zeros((x.shape[1], y.shape[1])),
                "bias": y_mean, "alpha": float("inf")}
    centered = x - mean
    scale = centered.std(0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    scaled = centered / scale
    y_scale = (y - y_mean).std(0)
    y_scale = np.where(y_scale < 1e-6, 1.0, y_scale)
    gram = scaled.T @ scaled + float(alpha) * np.eye(scaled.shape[1])
    weight_scaled = np.linalg.solve(gram, scaled.T @ ((y - y_mean) / y_scale))
    weight = weight_scaled * y_scale / scale[:, None]
    bias = y_mean - mean @ weight
    return {"mean": mean, "scale": scale, "weight": weight, "bias": bias, "alpha": float(alpha)}


def predict_ridge(features, model):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(model["mean"]):
        raise ValueError("features do not match the fitted ridge")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    scaled = (x - model["mean"]) / model["scale"]
    return scaled @ model["weight"] + model["bias"]


def select_ridge(features, targets, patients, alphas):
    """Choose a penalty by leave-one-patient-out, then refit on every training row."""
    x, y = _as_matrices(features, targets)
    if len(patients) != len(x):
        raise ValueError("patient ids must align with the training rows")
    grid = np.asarray(list(alphas), dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid) | np.isinf(grid)) or np.any(grid <= 0):
        raise ValueError("at least two positive penalties are required")
    folds = leave_one_patient_out(patients)
    scores = []
    for alpha in grid:
        errors = []
        for train_rows, held_rows in folds:
            model = fit_ridge(x[train_rows], y[train_rows], float(alpha))
            residual = predict_ridge(x[held_rows], model) - y[held_rows]
            errors.append(float((residual ** 2).sum(1).mean()))
        scores.append(float(np.mean(errors)))
    order = np.lexsort((-grid, np.asarray(scores)))
    chosen = float(grid[int(order[0])])
    model = fit_ridge(x, y, chosen)
    finite = grid[np.isfinite(grid)]
    model["cv_scores"] = scores
    model["alphas"] = [None if np.isinf(value) else float(value) for value in grid]
    model["intercept_only"] = bool(np.isinf(chosen))
    model["alpha_at_boundary"] = bool(
        np.isfinite(chosen) and (chosen == float(finite.min()) or chosen == float(finite.max())))
    return model
