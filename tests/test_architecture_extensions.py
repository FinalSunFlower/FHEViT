import unittest

import torch

from fpavlm.anatomical_slots import slot_token_groups
from fpavlm.cipher_native.demographic_filter import (
    LesionAwareDemNull,
    apply_lesion_aware_demographic_null_space,
)
from fpavlm.cipher_native.diagnostic_bottleneck import DiagnosticBottleneckRelease
from fpavlm.pipeline import FHEConfig


class ArchitectureExtensionTests(unittest.TestCase):
    def test_canonical_slots_partition_patch_grid(self):
        groups = slot_token_groups()
        self.assertEqual(len(groups), 5)
        self.assertTrue(all(group for group in groups))
        flattened = [index for group in groups for index in group]
        self.assertEqual(sorted(flattened), list(range(16)))

    def test_bottleneck_factory_exposes_five_spatial_releases(self):
        generator = torch.Generator().manual_seed(4)
        basis = torch.linalg.qr(torch.randn(8, 2, generator=generator)).Q
        release = DiagnosticBottleneckRelease.with_anatomical_slots(
            basis, torch.randn(8, 6, generator=generator)
        )
        values = release.encode_plain(torch.randn(3, 16, 8, generator=generator))
        self.assertEqual(tuple(values.shape), (3, 5, 2))
        self.assertEqual(release.release_token_count, 5)

    def test_anatomical_config_requires_matching_square_grid(self):
        config = FHEConfig(
            release_mode="diagnostic_bottleneck",
            dsr_calibration_path="calibration.pt",
            release_slot_mode="anatomical",
        )
        self.assertEqual(len(config.anatomical_release_groups()), 5)
        with self.assertRaisesRegex(ValueError, "square patch grid"):
            FHEConfig(
                release_slot_mode="anatomical",
                num_patches=15,
                grid_size=4,
                image_size=224,
                patch_size=56,
            )
        with self.assertRaisesRegex(ValueError, "release_mode"):
            FHEConfig(release_slot_mode="anatomical")

    def test_shallow_wide_preset_is_low_depth(self):
        config = FHEConfig.shallow_wide()
        self.assertEqual(config.num_layers, 2)
        self.assertFalse(config.use_cadn)
        self.assertLess(config._required_continuous_depth, 18)

    def test_lesion_aware_filter_preserves_orthogonality(self):
        generator = torch.Generator().manual_seed(3)
        features = torch.randn(40, 4, 8, generator=generator)
        labels = torch.arange(40) % 2
        lesion = torch.zeros(40, 4)
        lesion[:, 1:3] = 1.0
        basis, metrics = apply_lesion_aware_demographic_null_space(
            torch.linalg.qr(torch.randn(8, 3, generator=generator)).Q,
            features,
            lesion,
            labels,
            n_directions=1,
            method="mean_diff",
        )
        torch.testing.assert_close(basis.T @ basis, torch.eye(basis.shape[1]), atol=1e-5, rtol=1e-5)
        self.assertIn("lesion_energy_retention", metrics)
        fitted = LesionAwareDemNull.fit(features, labels, lesion, method="mean_diff", n_directions=1)
        self.assertEqual(fitted.demographic_directions.shape[0], 8)


if __name__ == "__main__":
    unittest.main()
