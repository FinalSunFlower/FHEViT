"""Unit tests for LBR/PBSR wiring and DemNull-FDBR."""
from __future__ import annotations

import unittest

import torch

from fpavlm.cipher_native.demographic_filter import (
    DemographicNullSpaceFilter,
    estimate_expected_attribute_auc_reduction,
    filter_release_residual,
)
from fpavlm.cipher_native.lbr_scheduler import default_lbr_schedule
from fpavlm.cipher_native.pbsr_scheduler import PBSRScheduler
from fpavlm.cipher_native.refresh_controller import (
    RefreshController,
    merge_ciphertexts,
    select_ciphertexts,
)


class RefreshControllerTests(unittest.TestCase):
    def test_default_controller_refreshes_every_block_and_all_tokens(self):
        controller = RefreshController(num_tokens=16)
        for block_idx in range(3):
            action = controller.action_before_block(block_idx)
            self.assertTrue(action.refresh)
            self.assertIsNone(action.token_indices)
            self.assertFalse(action.use_baked_norm)
        final = controller.action_final_norm()
        self.assertTrue(final.refresh)

    def test_lbr_skips_early_blocks_and_keeps_final_norm(self):
        schedule = default_lbr_schedule(
            num_blocks=3,
            use_cadn=True,
            poly_norm_degree=2,
            enable_token_mix=True,
            depth_budget=13,
            strategy="diagnostic_priority",
            cadn_fraction=0.5,
        )
        controller = RefreshController(schedule=schedule, num_tokens=16)
        first = controller.action_before_block(0)
        self.assertTrue(first.refresh)
        self.assertTrue(first.compute_block)
        skipped = controller.action_before_block(1)
        self.assertFalse(skipped.refresh)
        self.assertFalse(skipped.compute_block)
        self.assertTrue(controller.action_before_block(2).refresh)
        self.assertTrue(controller.action_before_block(2).compute_block)
        self.assertTrue(controller.action_final_norm().refresh)

    def test_pbsr_restricts_token_indices(self):
        mask = PBSRScheduler(num_tokens=16, grid_size=4).compute_mask()
        controller = RefreshController(mask=mask, num_tokens=16)
        action = controller.action_before_block(0)
        self.assertTrue(action.compute_block)
        self.assertEqual(action.token_indices, mask.diagnostic_indices)
        self.assertEqual(len(action.token_indices), 10)

    def test_combined_token_refresh_count_is_below_full_cadn(self):
        schedule = default_lbr_schedule(
            num_blocks=3,
            strategy="diagnostic_priority",
            cadn_fraction=0.5,
        )
        mask = PBSRScheduler(num_tokens=16, grid_size=4).compute_mask()
        controller = RefreshController(schedule=schedule, mask=mask, num_tokens=16)
        reduced = controller.expected_token_refreshes(num_blocks=3)
        full = RefreshController(num_tokens=16).expected_token_refreshes(num_blocks=3)
        self.assertLess(reduced, full)
        self.assertEqual(full, 4 * 16)

    def test_select_and_merge_ciphertexts(self):
        cipher_list = ["c0", "c1", "c2", "c3"]
        indices, selected = select_ciphertexts(cipher_list, (1, 3))
        self.assertEqual(selected, ["c1", "c3"])
        merged = merge_ciphertexts(cipher_list, indices, ["f1", "f3"])
        self.assertEqual(merged, ["c0", "f1", "c2", "f3"])


class DemographicNullSpaceTests(unittest.TestCase):
    def test_sex_direction_is_removed_from_basis(self):
        torch.manual_seed(7)
        n_samples, dim, rank = 80, 32, 4
        sex = torch.zeros(n_samples, dtype=torch.long)
        sex[n_samples // 2:] = 1
        gender = torch.zeros(dim)
        gender[0] = 1.0
        features = torch.randn(n_samples, dim)
        features[sex == 1] += 3.0 * gender

        rest = torch.randn(dim, rank)
        rest[0] = 0
        basis = torch.linalg.qr(rest)[0]
        basis = torch.nn.functional.normalize(basis + 0.4 * gender.unsqueeze(1), dim=0)

        filt = DemographicNullSpaceFilter.calibrate(
            features, sex_labels=sex, method="mean_diff", n_directions_per_attribute=1
        )
        filtered, metrics = filt.filter_subspace(basis, n_iter=3)
        overlap = torch.linalg.vector_norm(filt.demographic_directions.T @ filtered)
        self.assertLess(overlap.item(), 1e-4)
        self.assertGreater(metrics["overlap_reduction_pct"], 90.0)

    def test_auc_estimator_drops_below_gate(self):
        estimated = estimate_expected_attribute_auc_reduction(0.35, 0.07, 0.7035)
        self.assertLess(estimated, 0.60)
        self.assertGreater(estimated, 0.50)

    def test_release_residual_removes_membership_direction(self):
        torch.manual_seed(3)
        features = torch.randn(60, 16)
        membership = torch.zeros(60, dtype=torch.long)
        membership[30:] = 1
        leak = torch.zeros(16, 3)
        leak[0, 0] = 1.0
        leak[1, 1] = 1.0
        leak[2, 2] = 1.0
        features[membership == 1] += 4.0 * leak[:, 0]
        filtered, metrics = filter_release_residual(
            leak, features, (membership,), n_directions=1, n_iter=2
        )
        self.assertGreaterEqual(metrics["k_retained"], 1)
        release = features @ filtered
        score = release.mean(1)
        pos = score[membership == 1].mean()
        neg = score[membership == 0].mean()
        self.assertLess(abs(float(pos - neg)), 0.75)


if __name__ == "__main__":
    unittest.main()
