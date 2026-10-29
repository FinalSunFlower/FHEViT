"""Patient-disjoint diagnostic probes and paired non-inferiority intervals."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def diagnostic_probe(features, labels, patients, *, folds: int = 5) -> dict:
    """Fit preprocessing/classifiers on training patients; score out of fold.

    Features must already be frozen independently of evaluation patients for
    confirmatory use. This function cannot repair representation-training leaks.
    """
    x, y, groups = np.asarray(features), np.asarray(labels), np.asarray(patients)
    x = x.reshape(len(x), -1)
    if y.ndim != 2 or len(y) != len(x) or len(groups) != len(x):
        raise ValueError("features, multilabel targets, and patients must align")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("finite features and binary targets are required")
    count = min(folds, len(np.unique(groups)))
    if count < 2:
        raise ValueError("diagnostic probe needs at least two patient groups")
    predictions = np.zeros_like(y, dtype=float)
    splits = list(GroupKFold(count).split(x, groups=groups))
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise ValueError("patient overlap in diagnostic probe")
        for column in range(y.shape[1]):
            target = y[train, column]
            if len(np.unique(target)) < 2:
                # A missing training class cannot be learned. Report its
                # training prevalence, never fill in the held-out labels.
                predictions[test, column] = target.mean()
            else:
                probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1., random_state=42))
                probe.fit(x[train], target)
                predictions[test, column] = probe.predict_proba(x[test])[:, 1]
    valid = [i for i in range(y.shape[1]) if len(np.unique(y[:, i])) == 2]
    if not valid:
        raise ValueError("no diagnostic label has both classes")
    scores = [float(average_precision_score(y[:, i], predictions[:, i])) for i in valid]
    return {"macro_ap": float(np.mean(scores)), "per_label_ap": dict(zip(map(str, valid), scores)),
            "folds": count, "patients": len(np.unique(groups)), "protocol": "patient_disjoint_oof_logistic_v1"}


def paired_patient_noninferiority(native, candidate, patients, *, margin: float = .05,
                                 bootstrap_samples: int = 2000, seed: int = 42) -> dict:
    """Patient-weighted paired difference with a one-sided 95% bootstrap bound.

    Inputs are per-study clinical scores (higher is better) from the SAME
    patients/prompts. Margin is absolute, fixed before looking at test data.
    This assesses non-inferiority, never equality or absence of all loss.
    """
    left, right, groups = np.asarray(native, dtype=float), np.asarray(candidate, dtype=float), np.asarray(patients)
    if left.ndim != 1 or right.shape != left.shape or groups.shape != left.shape:
        raise ValueError("paired scores and patients must have identical one-dimensional shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all() or not 0 < margin < 1:
        raise ValueError("finite scores and a fixed margin in (0, 1) are required")
    ids = np.unique(groups)
    if len(ids) < 2 or bootstrap_samples < 100:
        raise ValueError("at least two patients and 100 bootstrap samples are required")
    differences = np.array([(right - left)[groups == p].mean() for p in ids])
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, (bootstrap_samples, len(ids)), replace=True).mean(1)
    lower = float(np.quantile(draws, .05))
    return {"patient_count": len(ids), "mean_difference": float(differences.mean()),
            "one_sided_95_lower": lower, "absolute_margin": margin, "noninferior": lower > -margin,
            "bootstrap_samples": bootstrap_samples, "seed": seed}


def paired_auc_noninferiority(labels, native_scores, release_scores, patients, *, margin: float = .05,
                             bootstrap_samples: int = 2000, seed: int = 42) -> dict:
    """Patient bootstrap of AUC(release) minus AUC(native) on one shared sample.

    Both scores are ranked on the same resampled patients. A draw that loses
    either class is skipped. The margin is absolute and fixed before the scores.
    """
    y = np.asarray(labels)
    native = np.asarray(native_scores, dtype=float)
    release = np.asarray(release_scores, dtype=float)
    groups = np.asarray(patients)
    if y.ndim != 1 or native.shape != y.shape or release.shape != y.shape or groups.shape != y.shape:
        raise ValueError("labels, both score columns, and patients must share one shape")
    if not np.isfinite(native).all() or not np.isfinite(release).all() or not 0 < margin < 1:
        raise ValueError("finite scores and a fixed margin in (0, 1) are required")
    point_native, point_release = roc_auc(y, native), roc_auc(y, release)
    ids = np.unique(groups)
    if point_native is None or point_release is None or len(ids) < 2 or bootstrap_samples < 100:
        raise ValueError("both classes, two patients, and 100 bootstrap draws are required")
    index_groups = {patient: np.flatnonzero(groups == patient) for patient in ids}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap_samples):
        chosen = rng.choice(ids, size=len(ids), replace=True)
        rows = np.concatenate([index_groups[patient] for patient in chosen])
        left, right = roc_auc(y[rows], native[rows]), roc_auc(y[rows], release[rows])
        if left is not None and right is not None:
            draws.append(right - left)
    if len(draws) < 100:
        raise ValueError("fewer than 100 bootstrap draws contained both classes")
    samples = np.asarray(draws, dtype=float)
    lower = float(np.quantile(samples, .05))
    return {"patient_count": int(len(ids)), "auc_native": point_native, "auc_release": point_release,
            "difference": point_release - point_native, "one_sided_95_lower": lower,
            "absolute_margin": margin, "noninferior": lower > -margin,
            "valid_draws": int(len(samples)), "bootstrap_samples": bootstrap_samples, "seed": seed}


def paired_metric_advantage(candidate, baseline, patients, *, bootstrap_samples: int = 2000,
                            seed: int = 42) -> dict:
    """Patient-level bootstrap interval for a per-study advantage over a baseline.

    ``candidate`` and ``baseline`` are per-study scores from the SAME studies where
    higher is better (for example the per-study feature MSE reduction relative to a
    constant-mean release, or per-study cosine). The paired advantage is aggregated
    per patient first, then patients are resampled. ``significant`` is only true when
    the two-sided 95% interval excludes zero; a point estimate alone is never enough
    to distinguish a real image-dependent signal from sampling noise. This measures a
    surrogate advantage, never clinical adequacy or report factuality.
    """
    left = np.asarray(candidate, dtype=float)
    right = np.asarray(baseline, dtype=float)
    groups = np.asarray(patients)
    if left.ndim != 1 or right.shape != left.shape or groups.shape != left.shape:
        raise ValueError("candidate, baseline and patients must share one-dimensional shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("finite candidate and baseline scores are required")
    ids = np.unique(groups)
    if len(ids) < 2 or bootstrap_samples < 100:
        raise ValueError("at least two patients and 100 bootstrap samples are required")
    advantages = np.array([(left - right)[groups == p].mean() for p in ids])
    rng = np.random.default_rng(seed)
    draws = rng.choice(advantages, (bootstrap_samples, len(ids)), replace=True).mean(1)
    ci_lower, ci_upper = float(np.quantile(draws, .025)), float(np.quantile(draws, .975))
    return {"patient_count": len(ids), "mean_advantage": float(advantages.mean()),
            "ci95_lower": ci_lower, "ci95_upper": ci_upper,
            "one_sided_95_lower": float(np.quantile(draws, .05)),
            "prob_positive": float((draws > 0).mean()),
            "significant_positive": ci_lower > 0, "significant_negative": ci_upper < 0,
            "bootstrap_samples": bootstrap_samples, "seed": seed}


def roc_auc(labels, scores):
    """AUC of one score column. Missing a class means the AUC is undefined."""
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.shape != y.shape:
        raise ValueError("labels and scores must share one-dimensional shapes")
    if not np.isfinite(s).all():
        raise ValueError("finite scores are required")
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def patient_mean_interval(scores, patients, *, bootstrap_samples: int = 2000, seed: int = 42) -> dict:
    """Patient-level bootstrap interval for a mean score. One class is allowed."""
    values, groups = np.asarray(scores, dtype=float), np.asarray(patients)
    if values.ndim != 1 or groups.shape != values.shape:
        raise ValueError("scores and patients must share one-dimensional shapes")
    if not np.isfinite(values).all():
        raise ValueError("finite scores are required")
    ids = np.unique(groups)
    if len(ids) < 2 or bootstrap_samples < 100:
        raise ValueError("at least two patients and 100 bootstrap samples are required")
    means = np.array([values[groups == patient].mean() for patient in ids])
    draws = np.random.default_rng(seed).choice(means, (bootstrap_samples, len(ids)), replace=True).mean(1)
    return {"patient_count": len(ids), "mean": float(means.mean()),
            "ci95_lower": float(np.quantile(draws, .025)), "ci95_upper": float(np.quantile(draws, .975)),
            "bootstrap_samples": bootstrap_samples, "seed": seed}


def patient_roc_auc_interval(labels, scores, patients, *, bootstrap_samples: int = 2000, seed: int = 42) -> dict:
    """Patient bootstrap of ROC AUC. A one-class resample is skipped, not filled in."""
    y, s, groups = np.asarray(labels), np.asarray(scores, dtype=float), np.asarray(patients)
    if y.ndim != 1 or s.shape != y.shape or groups.shape != y.shape:
        raise ValueError("labels, scores and patients must share one-dimensional shapes")
    point = roc_auc(y, s)
    ids = np.unique(groups)
    if point is None or len(ids) < 2 or bootstrap_samples < 100:
        return {"auc": point, "auc_defined": point is not None, "patient_count": int(len(ids)),
                "ci95_lower": None, "ci95_upper": None, "one_sided_95_lower": None,
                "above_chance": None, "valid_draws": 0, "bootstrap_samples": bootstrap_samples, "seed": seed}
    index_groups = {patient: np.flatnonzero(groups == patient) for patient in ids}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap_samples):
        chosen = rng.choice(ids, size=len(ids), replace=True)
        rows = np.concatenate([index_groups[patient] for patient in chosen])
        value = roc_auc(y[rows], s[rows])
        if value is not None:
            draws.append(value)
    if len(draws) < 100:
        raise ValueError("fewer than 100 bootstrap draws contained both classes")
    samples = np.asarray(draws, dtype=float)
    one_sided = float(np.quantile(samples, .05))
    return {"auc": point, "auc_defined": True, "patient_count": int(len(ids)),
            "ci95_lower": float(np.quantile(samples, .025)), "ci95_upper": float(np.quantile(samples, .975)),
            "one_sided_95_lower": one_sided, "above_chance": one_sided > 0.5,
            "valid_draws": int(len(samples)), "bootstrap_samples": bootstrap_samples, "seed": seed}
