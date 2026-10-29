import numpy as np

from fpavlm.evaluation import (
    diagnostic_probe,
    paired_auc_noninferiority,
    paired_metric_advantage,
    paired_patient_noninferiority,
    patient_mean_interval,
    patient_roc_auc_interval,
    roc_auc,
)


def test_auc_noninferiority_uses_one_shared_sample():
    labels = [1, 1, 1, 0, 0, 0]
    patients = [0, 1, 2, 3, 4, 5]
    native = [3., 2., 1., 0., -1., -2.]
    same = paired_auc_noninferiority(labels, native, native, patients)
    assert same["auc_native"] == same["auc_release"] == 1.
    assert same["difference"] == 0.
    assert same["noninferior"]
    reversed_scores = [-3., -2., -1., 0., 1., 2.]
    opposite = paired_auc_noninferiority(labels, native, reversed_scores, patients)
    assert opposite["auc_release"] == 0.
    assert opposite["noninferior"] is False


def test_noninferiority_counts_patients_not_repeated_studies():
    result = paired_patient_noninferiority([1, 1, 1, 1], [.8, .8, .8, 1], [0, 0, 0, 1], margin=.05)
    assert result["patient_count"] == 2
    assert abs(result["mean_difference"] + .1) < 1e-8
    assert not result["noninferior"]
    assert paired_patient_noninferiority([.8, .7], [.8, .7], [0, 1])["noninferior"]


def test_diagnostic_probe_learns_rotated_features():
    rng = np.random.default_rng(93)
    label = rng.integers(0, 2, size=(60, 1))
    # First coordinate is deliberately anticorrelated. Raw coordinate AP is
    # not a valid utility measure; a properly trained probe recovers the label.
    features = np.column_stack([-label[:, 0] * 10., rng.normal(size=60)])
    result = diagnostic_probe(features, label, np.repeat(np.arange(30), 2))
    assert result["macro_ap"] > .99
    assert result["patients"] == 30


def test_metric_advantage_flags_significance_by_patient_interval():
    rng = np.random.default_rng(7)
    patients = np.repeat(np.arange(40), 2)
    baseline = rng.normal(size=80)
    # A clearly positive per-study advantage must be flagged significant.
    clear = baseline + 1. + rng.normal(scale=.05, size=80)
    strong = paired_metric_advantage(clear, baseline, patients)
    assert strong["mean_advantage"] > .9
    assert strong["significant_positive"] and not strong["significant_negative"]
    assert strong["ci95_lower"] > 0
    # A near-zero advantage (the 0.3 situation) must NOT be called significant.
    tiny = baseline + rng.normal(scale=1., size=80) * 0. + rng.normal(scale=.5, size=80)
    weak = paired_metric_advantage(tiny, baseline, patients)
    assert not weak["significant_positive"]
    assert weak["ci95_lower"] < 0 < weak["ci95_upper"]


def test_auc_is_undefined_when_a_class_is_missing():
    assert roc_auc([1, 1, 1], [0.2, 0.8, 0.4]) is None
    assert roc_auc([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2]) == 1.0


def test_patient_auc_interval_is_undefined_for_one_class_and_high_when_separated():
    missing = patient_roc_auc_interval([1, 1, 1], [0.1, 0.2, 0.4], ["a", "b", "c"])
    assert missing["auc"] is None and missing["above_chance"] is None
    separated = patient_roc_auc_interval(
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], ["a", "b", "c", "d"], bootstrap_samples=200,
    )
    assert separated["auc"] == 1.0
    assert separated["one_sided_95_lower"] > 0.5
    assert separated["above_chance"] is True


def test_patient_mean_interval_uses_patients():
    result = patient_mean_interval([1.0, 1.0, 3.0], ["a", "a", "b"])
    assert result["patient_count"] == 2
    assert result["mean"] == 2.0
    assert result["ci95_lower"] <= 2.0 <= result["ci95_upper"]
