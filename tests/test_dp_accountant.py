import math
import unittest

import torch

from fpavlm.cipher_native.dp_accountant import (
    DPCFPAccountant,
    compute_dp_guarantee,
    l2_clip,
    quantize_int8,
    sigma_for_target_epsilon,
)


class DPAccountantTests(unittest.TestCase):
    def test_rejects_invalid_anisotropic_dp_contract(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            DPCFPAccountant(1.0, 12, anisotropy_lambda=-0.1)
        with self.assertRaisesRegex(ValueError, "must divide"):
            DPCFPAccountant(
                1.0,
                12,
                diagnostic_sensitivity=torch.ones(5),
                anisotropy_lambda=1.0,
            )
        accountant = DPCFPAccountant(
            1.0,
            12,
            diagnostic_sensitivity=torch.ones(4),
            anisotropy_lambda=1.0,
        )
        with self.assertRaisesRegex(ValueError, "channel dimension"):
            accountant.apply_and_account(torch.ones(1, 2, 6))

    def test_clips_complete_embedding_not_each_token(self):
        features = torch.ones(2, 4, 8)
        clipped = l2_clip(features, clip_norm=1.0)
        norms = clipped.reshape(2, -1).norm(dim=1)
        torch.testing.assert_close(norms, torch.ones(2))

    def test_epsilon_uses_per_coordinate_sigma(self):
        eps_small, _, _ = compute_dp_guarantee(0.12, dim=64)
        eps_large, _, _ = compute_dp_guarantee(0.12, dim=40960)
        self.assertAlmostEqual(eps_small, eps_large, places=12)
        self.assertGreater(eps_large, 100.0)

    def test_target_sigma_round_trip(self):
        sigma, sigma_l2, _ = sigma_for_target_epsilon(1.0, dim=40960)
        eps, _, reported_l2 = compute_dp_guarantee(sigma, dim=40960)
        self.assertAlmostEqual(eps, 1.0, places=9)
        self.assertAlmostEqual(sigma_l2, reported_l2, places=9)
        self.assertAlmostEqual(sigma_l2, sigma * math.sqrt(40960), places=9)

    def test_default_scale_is_not_data_dependent(self):
        accountant = DPCFPAccountant(
            noise_fraction=24.0,
            dim=32,
            clip_norm=1.0,
            quantize=False,
        )
        _, report_a = accountant.apply_and_account(torch.ones(1, 4, 8))
        _, report_b = accountant.apply_and_account(torch.full((1, 4, 8), 100.0))
        expected = 1.0 / math.sqrt(32)
        self.assertAlmostEqual(report_a.feature_std, expected)
        self.assertAlmostEqual(report_b.feature_std, expected)
        self.assertAlmostEqual(report_a.epsilon, report_b.epsilon)

    def test_accounted_quantization_uses_fixed_public_range(self):
        values = torch.tensor([[-2.0, -0.5, 0.5, 2.0]])
        quantized = quantize_int8(values, value_range=1.0)
        self.assertEqual(float(quantized.abs().max()), 1.0)
        accountant = DPCFPAccountant(
            noise_fraction=1.0, dim=4, clip_norm=1.0, quantize=True
        )
        _, report = accountant.apply_and_account(values)
        self.assertEqual(report.quantization_range, 1.0)
        self.assertAlmostEqual(report.composed_epsilon(3), 3.0 * report.epsilon)
        self.assertTrue(report.contains_image_information)


if __name__ == "__main__":
    unittest.main()
