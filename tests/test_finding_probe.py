"""Finding probe maps FDBR coefficients without changing the release."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from fpavlm.cipher_native.finding_probe import FindingProbe, fit_ridge


class FindingProbeTests(unittest.TestCase):
    def test_separates_two_classes(self):
        features = torch.tensor(
            [[-1.0, 0.0], [-0.8, 0.1], [0.9, 0.0], [1.1, -0.1]],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        weight, bias, mean = fit_ridge(features, labels, ridge=0.1)
        probe = FindingProbe(
            weight=weight,
            bias=bias,
            feature_mean=mean,
            classes=("normal", "opacity"),
            threshold=0.5,
            min_top1=0.3,
        )
        self.assertIn("normal", probe.predicted_labels(features[0]))
        self.assertIn("opacity", probe.predicted_labels(features[2]))
        text = probe.describe(features[2])
        self.assertIsNotNone(text)
        self.assertIn("opacity", text)

    def test_roundtrip(self):
        probe = FindingProbe(
            weight=torch.eye(2),
            bias=torch.zeros(2),
            feature_mean=torch.zeros(2),
            classes=("a", "b"),
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "probe.pt"
            probe.save(str(path))
            loaded = FindingProbe.load(str(path))
        torch.testing.assert_close(loaded.weight, probe.weight)
        self.assertEqual(loaded.classes, probe.classes)
