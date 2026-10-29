"""Public spatial canvas after pooled FDBR decode does not enlarge the release."""
from __future__ import annotations

import unittest

import torch

from fpavlm.cipher_native.medical_concept_adapter import MedicalConceptAdapter


class PublicSpatialPriorTests(unittest.TestCase):
    def _adapter(self) -> MedicalConceptAdapter:
        torch.manual_seed(3)
        adapter = MedicalConceptAdapter(
            dim=8,
            llm_dim=11,
            num_tokens=4,
            pool_type="perceiver_decoder",
            student_grid=2,
            target_grid=2,
            perceiver_name="identity",
        )
        basis, _ = torch.linalg.qr(torch.randn(8, 2))
        adapter.configure_diagnostic_bottleneck(
            basis, token_groups=[[0, 1, 2, 3]]
        )
        return adapter

    def test_pooled_decode_stays_identical_without_prior(self):
        adapter = self._adapter()
        coefficients = torch.randn(1, 1, 2)
        decoded = adapter.server_side_bottleneck_decode(coefficients)
        self.assertEqual(decoded.shape, (1, 4, 11))
        torch.testing.assert_close(decoded[:, 0], decoded[:, 1])
        torch.testing.assert_close(decoded[:, 0], decoded[:, 3])

    def test_prior_diversifies_tokens_without_changing_release(self):
        adapter = self._adapter()
        pos = torch.randn(1, 4, 8)
        adapter.configure_public_spatial_prior(pos, scale=0.5)
        self.assertIsNotNone(adapter._public_spatial_prior)
        coefficients = torch.randn(2, 1, 2)
        raw = adapter.diagnostic_bottleneck.decode_plain(coefficients)
        decorated = adapter.server_side_bottleneck_decode(coefficients)
        self.assertEqual(decorated.shape, raw.shape)
        self.assertGreater(
            float((decorated[:, 0] - decorated[:, 1]).pow(2).mean().sqrt()),
            1e-4,
        )
        # Mean over tokens stays the diagnostic decode (centered prior).
        torch.testing.assert_close(
            decorated.mean(dim=1), raw.mean(dim=1), rtol=1e-5, atol=1e-5
        )

    def test_identity_groups_skip_prior(self):
        adapter = self._adapter()
        adapter.configure_diagnostic_bottleneck(
            adapter.diagnostic_bottleneck.basis,
            token_groups=[[0], [1], [2], [3]],
        )
        adapter.configure_public_spatial_prior(torch.randn(1, 4, 8))
        self.assertIsNone(adapter._public_spatial_prior)


if __name__ == "__main__":
    unittest.main()
