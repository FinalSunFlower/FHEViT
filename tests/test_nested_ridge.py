import numpy as np

from fpavlm.nested_ridge import fit_ridge, leave_one_patient_out, predict_ridge, select_ridge


def test_leave_one_patient_out_keeps_a_patients_images_together():
    patients = np.array(["a", "a", "b", "c", "c"])
    folds = leave_one_patient_out(patients)
    assert len(folds) == 3
    for train, held in folds:
        assert set(patients[train]).isdisjoint(set(patients[held]))
        assert len(held) >= 1


def test_ties_keep_the_stronger_penalty():
    features = np.random.default_rng(0).normal(size=(8, 3))
    targets = np.ones((8, 1))
    patients = np.repeat(np.arange(4), 2)
    chosen = select_ridge(features, targets, patients, np.array([0.1, 1.0, 10.0]))
    assert chosen["alpha"] == 10.0


def test_patient_loo_ridge_beats_an_unregularized_fit():
    rng = np.random.default_rng(0)
    patients = np.repeat(np.arange(8), 3)
    features = rng.normal(size=(24, 80))
    targets = rng.normal(size=(24, 1))
    train, test = patients < 6, patients >= 6
    chosen = select_ridge(features[train], targets[train], patients[train], np.logspace(-4, 6, 13))
    regularized = predict_ridge(features[test], chosen)
    unregularized = predict_ridge(features[test], fit_ridge(features[train], targets[train], 1e-8))

    def mse(prediction):
        return float(((prediction - targets[test]) ** 2).mean())

    assert chosen["alpha"] > 1.0
    assert mse(regularized) < mse(unregularized)
