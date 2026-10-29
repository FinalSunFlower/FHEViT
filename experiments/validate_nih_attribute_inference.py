"""Attribute inference from noiseless InternVL2 oracle coordinates.

EVALUATION_PROTOCOL.md section 2.20. Development fits the PCA and the
attacker. Confirmation only scores them. No clinical rank is chosen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.fit_nih_oracle_bridge import (  # noqa: E402
    CONFIRMATION,
    DEVELOPMENT,
    _native,
    _records,
)
from experiments.nih_pneumonia import (  # noqa: E402
    CONFIRMATION_NEGATIVE,
    CONFIRMATION_POSITIVE,
    REPLICATION_NEGATIVE,
    REPLICATION_POSITIVE,
)
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402
from fpavlm.evaluation import roc_auc  # noqa: E402

BACKEND = "internvl2-2b"
RANKS = (4, 8, 16)
DRAWS = 2000
SEED = 42


def above_chance(lower):
    """One-sided 95% lower bound strictly above 0.5."""
    if not np.isfinite(lower):
        raise ValueError("the lower bound must be finite")
    return bool(lower > 0.5)


def beats_training_mean(lower):
    """Two-sided 95% lower bound of the MAE reduction strictly above zero."""
    if not np.isfinite(lower):
        raise ValueError("the lower bound must be finite")
    return bool(lower > 0)


def _targets(records):
    sex, view, age = [], [], []
    for row in records:
        if row["patient_gender"] not in {"F", "M"} or row["view_position"] not in {"PA", "AP"}:
            raise ValueError("sex must be F or M and view must be PA or AP")
        sex.append(1 if row["patient_gender"] == "F" else 0)
        view.append(1 if row["view_position"] == "PA" else 0)
        age.append(float(row["patient_age"]))
    return {
        "sex_female": np.asarray(sex, dtype=int),
        "view_pa": np.asarray(view, dtype=int),
        "age": np.asarray(age, dtype=float),
    }


def _fit_binary(train_x, train_y, test_x):
    if len(np.unique(train_y)) < 2:
        raise ValueError("the development attribute has one class")
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(scaler.transform(train_x), train_y)
    if int(np.max(model.n_iter_)) >= 1000:
        raise RuntimeError("logistic attacker did not converge")
    return model.predict_proba(scaler.transform(test_x))[:, 1]


def _fit_age(train_x, train_y, test_x):
    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=1.0)
    model.fit(scaler.transform(train_x), train_y)
    return model.predict(scaler.transform(test_x)), float(train_y.mean())


def _binary_interval(labels, scores, draws):
    point = roc_auc(labels, scores)
    if point is None:
        raise ValueError("confirmation attribute AUC is undefined")
    samples = []
    for rows in draws:
        value = roc_auc(labels[rows], scores[rows])
        if value is not None:
            samples.append(value)
    if len(samples) < 100:
        raise ValueError("fewer than 100 bootstrap draws contained both classes")
    samples = np.asarray(samples, dtype=float)
    lower = float(np.quantile(samples, 0.05))
    ci_low, ci_high = (float(v) for v in np.quantile(samples, [0.025, 0.975]))
    return {
        "auc": point,
        "one_sided_95_lower": lower,
        "ci95_lower": ci_low,
        "ci95_upper": ci_high,
        "above_chance": above_chance(lower),
        "valid_draws": int(len(samples)),
        "samples": samples,
    }


def _age_interval(ages, predicted, baseline, rows):
    def mae(chosen, forecast):
        return float(np.mean(np.abs(ages[chosen] - forecast[chosen])))

    point_model = mae(np.arange(len(ages)), predicted)
    point_base = mae(np.arange(len(ages)), np.full(len(ages), baseline))
    reductions = np.asarray([
        mae(chosen, np.full(len(ages), baseline)) - mae(chosen, predicted) for chosen in rows
    ], dtype=float)
    lower, upper = (float(v) for v in np.quantile(reductions, [0.025, 0.975]))
    return {
        "mae": point_model,
        "baseline_mae": point_base,
        "mae_reduction": point_base - point_model,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "beats_training_mean": beats_training_mean(lower),
        "valid_draws": int(len(reductions)),
    }


def _valid_draws(labels, count, rng):
    kept = []
    for _ in range(DRAWS):
        rows = rng.choice(count, size=count, replace=True)
        if len(np.unique(labels[rows])) == 2:
            kept.append(rows)
    if len(kept) < 100:
        raise ValueError("fewer than 100 bootstrap draws contained both classes")
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=[BACKEND], default=BACKEND)
    args = parser.parse_args()
    if args.backend != BACKEND:
        raise SystemExit("section 2.20 fits internvl2-2b only")
    torch.set_num_threads(4)
    torch.manual_seed(812)
    started = time.perf_counter()
    development = _records(DEVELOPMENT, REPLICATION_POSITIVE + REPLICATION_NEGATIVE)
    confirmation = _records(CONFIRMATION, CONFIRMATION_POSITIVE + CONFIRMATION_NEGATIVE)
    if set(row["patient_id"] for row in development) & set(row["patient_id"] for row in confirmation):
        raise ValueError("development and confirmation patients overlap")
    train_targets = _targets(development)
    test_targets = _targets(confirmation)
    llm = load_backend(BACKEND)
    native = torch.cat([
        _native(llm, DEVELOPMENT, development, BACKEND),
        _native(llm, CONFIRMATION, confirmation, BACKEND),
    ]).float()
    train_n = len(development)
    centered = native.reshape(len(native), -1).double() - native[:train_n].reshape(train_n, -1).double().mean(0)
    basis = torch.linalg.svd(centered[:train_n], full_matrices=False).Vh.T
    pooled = native.mean(1).numpy()
    features = {"pooled": pooled}
    for rank in RANKS:
        features[f"pca_rank_{rank}"] = (centered @ basis[:, :rank]).float().numpy()
    binary = ("sex_female", "view_pa")
    predictions = {name: {} for name in (*binary, "age")}
    age_baseline = None
    for name, matrix in features.items():
        train_x, test_x = matrix[:train_n], matrix[train_n:]
        for attribute in binary:
            predictions[attribute][name] = _fit_binary(train_x, train_targets[attribute], test_x)
        predicted, age_baseline = _fit_age(train_x, train_targets["age"], test_x)
        predictions["age"][name] = predicted
    count = len(confirmation)
    rng = np.random.default_rng(SEED)
    sex_draws = _valid_draws(test_targets["sex_female"], count, rng)
    view_draws = _valid_draws(test_targets["view_pa"], count, rng)
    age_draws = [rng.choice(count, size=count, replace=True) for _ in range(DRAWS)]
    draw_bank = {"sex_female": sex_draws, "view_pa": view_draws}
    rows = []
    binary_samples = {}
    for attribute in binary:
        labels = test_targets[attribute]
        scored = {}
        for name in features:
            scored[name] = _binary_interval(labels, predictions[attribute][name], draw_bank[attribute])
            binary_samples[(attribute, name)] = scored[name].pop("samples")
        contrasts = []
        pooled_samples = binary_samples[(attribute, "pooled")]
        for rank in RANKS:
            delta = binary_samples[(attribute, f"pca_rank_{rank}")] - pooled_samples
            lower, upper = (float(v) for v in np.quantile(delta, [0.025, 0.975]))
            contrasts.append({
                "rank": rank,
                "auc_coordinates_minus_pooled": scored[f"pca_rank_{rank}"]["auc"] - scored["pooled"]["auc"],
                "ci95_lower": lower,
                "ci95_upper": upper,
                "includes_zero": lower <= 0 <= upper,
            })
        rows.append({"attribute": attribute, "features": scored, "versus_pooled": contrasts})
    age_rows = {"attribute": "age", "features": {}, "versus_pooled": []}
    for name in features:
        age_rows["features"][name] = _age_interval(
            test_targets["age"], predictions["age"][name], age_baseline, age_draws,
        )
    for rank in RANKS:
        coordinate = predictions["age"][f"pca_rank_{rank}"]
        control = predictions["age"]["pooled"]
        ages = test_targets["age"]
        gaps = np.asarray([
            float(np.mean(np.abs(ages[chosen] - control[chosen])) - np.mean(np.abs(ages[chosen] - coordinate[chosen])))
            for chosen in age_draws
        ])
        lower, upper = (float(v) for v in np.quantile(gaps, [0.025, 0.975]))
        age_rows["versus_pooled"].append({
            "rank": rank,
            "mae_pooled_minus_coordinates": (
                age_rows["features"]["pooled"]["mae"] - age_rows["features"][f"pca_rank_{rank}"]["mae"]
            ),
            "ci95_lower": lower,
            "ci95_upper": upper,
            "includes_zero": lower <= 0 <= upper,
        })
    result = {
        "schema": "nih_attribute_inference_v1",
        "protocol_section": "2.20",
        "backend": BACKEND,
        "development_patients": train_n,
        "confirmation_patients": count,
        "fit_saw_confirmation": False,
        "noiseless_coordinates": True,
        "ranks": list(RANKS),
        "control": "mean of the 256 native visual tokens",
        "attacks": rows + [age_rows],
        "development_mean_age": age_baseline,
        "clinical_rank_selected": False,
        "noninferiority_claim": False,
        "reconstruction_attack_started": False,
        "predictor_retarget_started": False,
        "mlp_started": False,
        "privacy_channel_redesign_started": False,
        "official_test_list_read": False,
        "limitations": [
            "Noiseless coordinates are an upper bound on what a noisy release of the same coordinates could leak.",
            "The attacker is fit once. The interval resamples confirmation predictions and does not refit.",
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    destination = ROOT / "experiments" / "results" / "nih_attribute_inference_internvl2-2b.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    printable = []
    for attack in result["attacks"]:
        if attack["attribute"] == "age":
            printable.append({
                "attribute": "age",
                "features": {
                    name: {"mae": row["mae"], "beats_training_mean": row["beats_training_mean"]}
                    for name, row in attack["features"].items()
                },
            })
        else:
            printable.append({
                "attribute": attack["attribute"],
                "features": {
                    name: {"auc": row["auc"], "one_sided_95_lower": row["one_sided_95_lower"],
                           "above_chance": row["above_chance"]}
                    for name, row in attack["features"].items()
                },
            })
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
