import tempfile
import unittest
from pathlib import Path

import torch

from fpavlm.deployment import DeploymentProfile
from fpavlm.privacy_audit import (
    PatientLevelPrivacyAuditor,
    PrivacyAuditThresholds,
    bind_privacy_audit_to_calibration,
    validate_privacy_audit,
    validate_release_calibration,
)
from fpavlm.vision_plugin import FHEVisionCapabilities, FHEVisionRegistry


class _PluginProbe:
    concept_adapter = object()

    def patchify(self):
        return None

    def forward(self):
        return None

    def bake_statistics(self):
        return None

    def multiplicative_depth(self):
        return 4

    def continuous_multiplicative_depth(self):
        return 21


def _attack_fixture(leaky: bool):
    generator = torch.Generator().manual_seed(13)
    patients = torch.arange(20).repeat_interleave(10)
    attributes = (patients % 2).long()
    membership = ((patients // 2) % 2).long()
    patient_code = torch.nn.functional.one_hot(patients, num_classes=20).float()
    targets = torch.randn(200, 8, generator=generator)
    if leaky:
        releases = torch.cat(
            [
                10.0 * patient_code,
                10.0 * attributes[:, None].float(),
                10.0 * membership[:, None].float(),
                targets,
            ],
            dim=1,
        )
    else:
        releases = torch.randn(200, 8, generator=generator)
    return releases, targets, patients, attributes, membership


class PrivacyAuditTests(unittest.TestCase):
    @staticmethod
    def _evidence_scope(report, **overrides):
        value = {
            "status": "external_confirmation",
            "external_confirmation": True,
            "attack_protocol": "five_fold_patient_disjoint_out_of_fold",
            "attack_sample_count": report.sample_count,
            "attack_patient_count": report.patient_count,
            "privacy_cohort_used_for_adversarial_training": False,
        }
        value.update(overrides)
        return value

    def test_leaky_release_fails_multiple_attacks(self):
        report = PatientLevelPrivacyAuditor().evaluate(*_attack_fixture(leaky=True))
        self.assertFalse(report.passed)
        self.assertLess(report.reconstruction_nrmse, 0.90)
        self.assertGreater(report.patient_linkage_rate, 0.90)
        self.assertGreater(report.attribute_auc, 0.90)
        self.assertGreater(report.membership_auc, 0.90)

    def test_random_release_can_pass_promotion_gate(self):
        report = PatientLevelPrivacyAuditor().evaluate(*_attack_fixture(leaky=False))
        self.assertTrue(report.passed)
        validate_privacy_audit({"privacy_audit": report.to_dict()})

    def test_forged_pass_flag_is_rejected(self):
        report = PatientLevelPrivacyAuditor().evaluate(*_attack_fixture(leaky=True))
        forged = report.to_dict()
        forged["passed"] = True
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_privacy_audit({"privacy_audit": forged})

    def test_weaker_or_malformed_audit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "weaker"):
            PatientLevelPrivacyAuditor(
                PrivacyAuditThresholds(max_attribute_auc=0.90)
            )
        report = PatientLevelPrivacyAuditor().evaluate(*_attack_fixture(leaky=False))
        malformed = report.to_dict()
        malformed["attribute_auc"] = -0.1
        with self.assertRaisesRegex(ValueError, "valid ranges"):
            validate_privacy_audit({"privacy_audit": malformed})
        malformed = report.to_dict()
        malformed["sample_count"] = 200.5
        with self.assertRaisesRegex(ValueError, "positive integers"):
            validate_privacy_audit({"privacy_audit": malformed})
        malformed = report.to_dict()
        malformed["passed"] = "true"
        with self.assertRaisesRegex(ValueError, "boolean"):
            validate_privacy_audit({"privacy_audit": malformed})

    def test_infeasible_calibration_cannot_be_promoted(self):
        report = PatientLevelPrivacyAuditor().evaluate(*_attack_fixture(leaky=False))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            target = Path(directory) / "target.pt"
            torch.save(
                {
                    "projection": torch.eye(2),
                    "subspace": torch.eye(2),
                    "k": 2,
                    "label_classes": ["finding"],
                    "metrics": {"selected_candidate_feasible": False},
                },
                source,
            )
            with self.assertRaisesRegex(ValueError, "utility/linkage"):
                bind_privacy_audit_to_calibration(str(source), report, str(target))
            self.assertFalse(target.exists())

    def test_secure_release_requires_both_promotion_gates(self):
        with self.assertRaisesRegex(ValueError, "selected_candidate_feasible"):
            validate_release_calibration({}, require_privacy_audit=True)
        with self.assertRaisesRegex(ValueError, "privacy_audit"):
            validate_release_calibration(
                {"selected_candidate_feasible": True},
                require_privacy_audit=True,
            )

    def test_secure_release_binds_the_exact_audited_mechanism(self):
        report = PatientLevelPrivacyAuditor().evaluate(*_attack_fixture(leaky=False))
        metrics = {
            "selected_candidate_feasible": True,
            "privacy_audit": report.to_dict(),
            "evidence_scope": self._evidence_scope(report),
            "mechanism": {
                "whole_record_clip_norm": 2.5,
                "target_epsilon": 10.0,
                "release_dim": 2,
            },
        }
        validate_release_calibration(
            metrics,
            require_privacy_audit=True,
            expected_mechanism={
                "whole_record_clip_norm": 2.5,
                "target_epsilon": 10.0,
                "release_dim": 2,
            },
        )
        with self.assertRaisesRegex(ValueError, "target_epsilon.*mismatch"):
            validate_release_calibration(
                metrics,
                require_privacy_audit=True,
                expected_mechanism={"target_epsilon": 3.0},
            )

    def test_secure_release_requires_consistent_evidence_scope(self):
        report = PatientLevelPrivacyAuditor().evaluate(*_attack_fixture(leaky=False))
        metrics = {
            "selected_candidate_feasible": True,
            "privacy_audit": report.to_dict(),
        }
        with self.assertRaisesRegex(ValueError, "evidence_scope"):
            validate_release_calibration(metrics, require_privacy_audit=True)
        metrics["evidence_scope"] = self._evidence_scope(
            report, attack_patient_count=report.patient_count + 1
        )
        with self.assertRaisesRegex(ValueError, "counts do not match"):
            validate_release_calibration(metrics, require_privacy_audit=True)

    def test_secure_release_rejects_development_only_evidence(self):
        report = PatientLevelPrivacyAuditor().evaluate(*_attack_fixture(leaky=False))
        metrics = {
            "selected_candidate_feasible": True,
            "privacy_audit": report.to_dict(),
            "evidence_scope": self._evidence_scope(
                report,
                status="development_feasibility",
                external_confirmation=False,
                privacy_cohort_used_for_adversarial_training=True,
            ),
        }
        with self.assertRaisesRegex(ValueError, "independent external confirmation"):
            validate_release_calibration(metrics, require_privacy_audit=True)

    def test_research_infeasible_release_requires_explicit_allow(self):
        metrics = {"selected_candidate_feasible": False}
        with self.assertRaisesRegex(ValueError, "marked infeasible"):
            validate_release_calibration(metrics, require_privacy_audit=False)
        validate_release_calibration(
            metrics, require_privacy_audit=False, allow_infeasible=True
        )
        with self.assertRaisesRegex(ValueError, "selected_candidate_feasible"):
            validate_release_calibration(
                metrics, require_privacy_audit=True, allow_infeasible=True
            )

    def test_audit_rejects_patient_split_without_class_coverage(self):
        releases, targets, patients, attributes, membership = _attack_fixture(
            leaky=False
        )
        attributes = (patients == 0).long()
        with self.assertRaisesRegex(ValueError, "no patient-disjoint split"):
            PatientLevelPrivacyAuditor().evaluate(
                releases, targets, patients, attributes, membership
            )


class VisionPluginTests(unittest.TestCase):
    def test_callback_free_capability_requires_raw_patches(self):
        with self.assertRaisesRegex(ValueError, "raw patch"):
            FHEVisionCapabilities(
                input_domain="projected_patches",
                output_domain="diagnostic_coefficients",
                homomorphic_operations=("add", "cipher_plain_mul"),
                supports_interactive=True,
                supports_callback_free=True,
                protects_server_weights_callback_free=True,
                supports_raw_patch_ciphertext=False,
            )

    def test_registered_plugin_depth_is_validated(self):
        name = "unit-test-depth-probe"
        capabilities = FHEVisionCapabilities(
            input_domain="raw_patches",
            output_domain="diagnostic_coefficients",
            homomorphic_operations=("add", "cipher_plain_mul"),
            supports_interactive=False,
            supports_callback_free=True,
            protects_server_weights_callback_free=True,
            supports_raw_patch_ciphertext=True,
        )
        if name not in FHEVisionRegistry.available():
            FHEVisionRegistry.register(name, capabilities=capabilities)(_PluginProbe)
        plugin = FHEVisionRegistry.load(
            name,
            deployment_profile="model_confidential",
        )
        with self.assertRaisesRegex(ValueError, "expected 20"):
            FHEVisionRegistry.validate_instance(
                plugin,
                profile=DeploymentProfile.MODEL_CONFIDENTIAL,
                expected_continuous_depth=20,
            )


if __name__ == "__main__":
    unittest.main()
