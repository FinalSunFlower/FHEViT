import os
import tempfile
import unittest
from pathlib import Path

import torch

from fpavlm.checkpoint import load_artifact_manifest, load_tensor_checkpoint
from fpavlm.cipher_native.ciphertext_tensor import CiphertextTensor
from fpavlm.cipher_native.diagnostic_bottleneck import DiagnosticBottleneckRelease
from fpavlm.cipher_native.dp_accountant import DPCFPAccountant
from fpavlm.cipher_native.noise_resilient_encoder import NoiseResilientMAViTEncoder
from fpavlm.cipher_native.subspace_defense import DiagnosticSubspaceRetention
from fpavlm.deployment import continuous_fhe_depth, validate_ckks_budget
from fpavlm.embedding_contract import DiagnosticEmbeddingRelease, LLMVisualEmbedding
from fpavlm.llm.interface import LLMInterface
from fpavlm.noise_calibration import FHENoiseCalibration
from fpavlm.pipeline import FHEConfig


class _TrustProbe(LLMInterface):
    def load(self):
        raise NotImplementedError

    def embed_image_tokens(self, image_embeds, prompt=None):
        raise NotImplementedError

    def generate(self, image_embeds, **kwargs):
        raise NotImplementedError

    def image_token_id(self):
        raise NotImplementedError

    def expected_image_token_count(self):
        return 256

    @property
    def llm_hidden_dim(self):
        return 8


class _DepthCipher:
    """Minimal ciphertext-shaped object for circuit-control-flow tests."""

    def __add__(self, other):
        return _DepthCipher()

    __radd__ = __add__

    def __sub__(self, other):
        return _DepthCipher()

    def __neg__(self):
        return _DepthCipher()

    def __mul__(self, other):
        return _DepthCipher()

    __rmul__ = __mul__

    def matmul(self, weight):
        return _DepthCipher()

    def polyval(self, coefficients):
        return _DepthCipher()


class _DepthEngine:
    def matmul_plain(self, ciphertexts, weight):
        return [ciphertext.matmul(weight) for ciphertext in ciphertexts]


class DiagnosticBottleneckTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(7)
        basis, _ = torch.linalg.qr(torch.randn(8, 3, generator=generator))
        self.basis = basis
        self.weight = torch.randn(8, 11, generator=generator)
        self.bias = torch.randn(11, generator=generator)
        self.release = DiagnosticBottleneckRelease(
            basis=self.basis,
            output_weight=self.weight,
            output_bias=self.bias,
        )

    def test_factorization_matches_projected_full_output(self):
        features = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(11))
        actual = self.release.projected_reference(features)
        projection = self.basis @ self.basis.T
        expected = features @ projection @ self.weight + self.bias
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_token_group_pooling_reduces_release_and_expands_privately(self):
        grouped = DiagnosticBottleneckRelease(
            basis=self.basis,
            output_weight=self.weight,
            output_bias=self.bias,
            token_groups=[[0, 1], [2, 3]],
        )
        features = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(13))
        coefficients = features @ self.basis
        released = grouped.encode_plain(features)
        expected_release = torch.stack(
            [coefficients[:, :2].mean(1), coefficients[:, 2:].mean(1)], dim=1
        )
        torch.testing.assert_close(released, expected_release)
        decoded = grouped.decode_plain(released)
        self.assertEqual(decoded.shape, (2, 4, 11))
        torch.testing.assert_close(decoded[:, 0], decoded[:, 1])
        torch.testing.assert_close(decoded[:, 2], decoded[:, 3])

        encrypted = CiphertextTensor(
            ciphertexts=[_DepthCipher() for _ in range(4)],
            engine=_DepthEngine(),
        )
        cipher_release = grouped.encode_cipher(encrypted)
        self.assertEqual(len(cipher_release._cipher), 2)
        self.assertEqual(cipher_release.depth, 2)

    def test_rejects_non_orthogonal_basis(self):
        bad_basis = self.basis.clone()
        bad_basis[:, 0] *= 2
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            DiagnosticBottleneckRelease(bad_basis, self.weight, self.bias)

    def test_target_epsilon_controls_accountant(self):
        accountant = DPCFPAccountant(
            noise_fraction=0.0,
            dim=12,
            target_epsilon=2.0,
            clip_norm=1.0,
            quantize=False,
        )
        _, report = accountant.apply_and_account(torch.zeros(1, 4, 3))
        self.assertAlmostEqual(report.epsilon, 2.0, places=8)

    def test_embedding_boundary_contracts_reject_shape_mismatch(self):
        accountant = DPCFPAccountant(
            noise_fraction=0.0,
            dim=12,
            target_epsilon=2.0,
            clip_norm=1.0,
        )
        values, report = accountant.apply_and_account(torch.zeros(1, 4, 3))
        release = DiagnosticEmbeddingRelease(
            values, report, "research_baseline", calibrated=False
        )
        self.assertEqual(release.values_per_image, 12)
        visual = LLMVisualEmbedding(torch.zeros(1, 4, 8), 3, "probe")
        with self.assertRaisesRegex(ValueError, "expects 256 image tokens"):
            visual.validate_for(_TrustProbe("unused"))

    def test_anatomical_compression_ratio_counts_all_released_slots(self):
        generator = torch.Generator().manual_seed(11)
        basis = torch.linalg.qr(torch.randn(8, 2, generator=generator)).Q
        release = DiagnosticBottleneckRelease.with_anatomical_slots(
            basis, torch.randn(8, 6, generator=generator)
        )
        self.assertEqual(release.metadata.released_values_per_image, 10)
        self.assertEqual(release.metadata.input_values_per_image, 16 * 6)
        self.assertAlmostEqual(release.metadata.compression_ratio, 10 / 96)

    def test_secure_release_requires_calibration(self):
        with self.assertRaisesRegex(ValueError, "requires dsr_calibration_path"):
            FHEConfig(
                release_mode="diagnostic_bottleneck",
                dsr_calibration_path=None,
                use_mafo_scale=False,
            )

    def test_rank_selection_uses_held_out_folds(self):
        generator = torch.Generator().manual_seed(19)
        diagnosis = torch.randint(0, 2, (120, 2), generator=generator).float()
        features = torch.cat(
            [diagnosis * 3 + 0.1 * torch.randn(120, 2, generator=generator),
             torch.randn(120, 6, generator=generator)],
            dim=1,
        )
        rank, results = DiagnosticSubspaceRetention.cross_validated_rank_selection(
            features,
            diagnosis.numpy(),
            candidate_ranks=(2, 4, 8),
            n_splits=4,
            max_linkage_rate=1.0,
            min_utility_retention=0.85,
        )
        self.assertEqual(rank, 2)
        selected = next(item for item in results if item["rank"] == rank)
        self.assertGreater(selected["utility_retention"], 0.85)

    def test_deployment_profiles_reject_inconsistent_contracts(self):
        with self.assertRaisesRegex(ValueError, "explicit cfp_target_epsilon"):
            FHEConfig(
                deployment_profile="utility_interactive",
                release_mode="diagnostic_bottleneck",
                dsr_calibration_path="calibration.pt",
                use_mafo_scale=False,
            )
        with self.assertRaisesRegex(ValueError, "use_cadn=False"):
            FHEConfig(
                deployment_profile="model_confidential",
                release_mode="diagnostic_bottleneck",
                dsr_calibration_path="calibration.pt",
                cfp_target_epsilon=3.0,
                use_mafo_scale=False,
                artifact_manifest_path="artifacts.json",
            )

    def test_continuous_depth_and_ckks_budget(self):
        depth = continuous_fhe_depth([1, 1, 1], token_mixing=True)
        self.assertEqual(depth, 21)
        budget = validate_ckks_budget(32768, 24, depth)
        self.assertEqual(budget.coefficient_modulus_bits, 624)
        with self.assertRaisesRegex(ValueError, "tc128 limit"):
            validate_ckks_budget(16384, 24, depth)
        linear_depth = continuous_fhe_depth(
            [1, 1, 1], token_mixing=True, poly_norm_degree=1
        )
        self.assertEqual(linear_depth, 18)
        self.assertEqual(validate_ckks_budget(32768, 24, linear_depth).coefficient_modulus_bits, 552)

    def test_callback_free_algebraic_fusions_reduce_depth_without_plain_gap(self):
        common = dict(
            dim=4,
            num_layers=1,
            mlp_ratio=1.0,
            num_patches=1,
            in_channels=1,
            patch_size=2,
            img_size=2,
            block_type="med_fhe_vit_v3",
            poly_act_degree=1,
            poly_norm_degree=1,
            num_regions=1,
            llm_dim=6,
            num_concepts=1,
            pool_type="direct",
            student_grid=1,
            target_grid=1,
            enable_token_mix=False,
            use_cadn=False,
            enable_noise_training=False,
        )
        torch.manual_seed(19)
        baseline = NoiseResilientMAViTEncoder(
            **common, fuse_linear_poly_act=False, fuse_final_affine=False
        ).eval()
        optimized = NoiseResilientMAViTEncoder(
            **common,
            fuse_linear_poly_act=True,
            fuse_linear_poly_norm=True,
            fuse_final_affine=True,
        ).eval()
        optimized.load_state_dict(baseline.state_dict())
        baseline.bake_statistics()
        optimized.bake_statistics()
        image = torch.randn(1, 1, 2, 2)
        patches = baseline.patchify(image)
        with torch.no_grad():
            expected = baseline.forward_plain_fhe_match(patches)
            actual = optimized.forward_plain_fhe_match(patches)
        self.assertLess(float((expected - actual).abs().max()), 4.0)
        self.assertGreater(
            float(torch.nn.functional.cosine_similarity(expected.flatten(), actual.flatten(), dim=0)),
            0.999999,
        )
        self.assertEqual(baseline.continuous_multiplicative_depth(), 7)
        self.assertEqual(optimized.continuous_multiplicative_depth(), 4)

    def test_callback_free_encoder_control_flow_reaches_depth_21(self):
        encoder = NoiseResilientMAViTEncoder(
            dim=4,
            num_layers=3,
            mlp_ratio=2.0,
            num_patches=4,
            in_channels=3,
            patch_size=1,
            img_size=2,
            block_type="med_fhe_vit_v3",
            poly_act_degree_schedule=[1, 1, 1],
            num_regions=5,
            llm_dim=6,
            num_concepts=1,
            pool_type="direct",
            student_grid=2,
            target_grid=2,
            enable_token_mix=True,
            use_cadn=False,
            enable_noise_training=False,
        ).eval()
        encoder.bake_statistics()
        basis, _ = torch.linalg.qr(torch.randn(4, 2))
        encoder.concept_adapter.configure_diagnostic_bottleneck(basis)
        encrypted = CiphertextTensor(
            ciphertexts=[_DepthCipher() for _ in range(4)],
            engine=_DepthEngine(),
        )
        output = encoder(
            encrypted,
            cspe_mode=False,
            refresh_policy="none",
        )
        self.assertTrue(output.is_cipher)
        self.assertEqual(output.depth, 21)
        self.assertEqual(len(output._cipher), 4)

    def test_model_confidential_profile_accepts_explicit_budget(self):
        config = FHEConfig(
            deployment_profile="model_confidential",
            release_mode="diagnostic_bottleneck",
            dsr_calibration_path="calibration.pt",
            cfp_target_epsilon=3.0,
            use_cadn=False,
            pre_cadn_noise_sigma=0.0,
            ckks_scale_bits=24,
            ckks_depth_budget=21,
            use_mafo_scale=False,
            artifact_manifest_path="artifacts.json",
        )
        self.assertEqual(config._required_continuous_depth, 18)

    def test_model_confidential_rejects_interactive_masking(self):
        with self.assertRaisesRegex(ValueError, "pre_cadn_noise_sigma must be 0"):
            FHEConfig(
                deployment_profile="model_confidential",
                release_mode="diagnostic_bottleneck",
                dsr_calibration_path="calibration.pt",
                cfp_target_epsilon=3.0,
                use_cadn=False,
                pre_cadn_noise_sigma=0.01,
                ckks_scale_bits=24,
                ckks_depth_budget=21,
                use_mafo_scale=False,
                artifact_manifest_path="artifacts.json",
            )

    def test_manifest_paths_are_relative_to_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "weights.pt"
            artifact.write_bytes(b"checkpoint")
            import hashlib
            import json
            digest = hashlib.sha256(b"checkpoint").hexdigest()
            manifest = root / "artifacts.json"
            manifest.write_text(
                json.dumps({"artifacts": {"weights.pt": digest}}),
                encoding="utf-8",
            )
            loaded = load_artifact_manifest(str(manifest))
            self.assertEqual(
                loaded[os.path.normcase(str(artifact.resolve()))], digest
            )

    def test_remote_code_requires_matching_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "modeling_custom.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            probe = _TrustProbe(
                directory,
                allow_remote_code=True,
                remote_code_sha256="0" * 64,
            )
            with self.assertRaisesRegex(PermissionError, "digest mismatch"):
                probe.verify_remote_code_trust()

    def test_noise_aware_decoding_requires_paired_calibration(self):
        with self.assertRaisesRegex(ValueError, "noise_calibration_path"):
            FHEConfig(use_cnad_decoding=True, use_mafo_scale=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noise.json"
            path.write_text(
                '{"method":"fresh_reencryption","sample_count":10,'
                '"embedding_error_std":1e-5,"embedding_rmse":1e-5}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "matching plaintext circuit"):
                FHENoiseCalibration.load(str(path))

    def test_checkpoint_digest_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"weight": torch.ones(1)}, path)
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_tensor_checkpoint(str(path), expected_sha256="0" * 64)

if __name__ == "__main__":
    unittest.main()
