"""Patient-level attack audit and fail-closed release-promotion contract."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from fpavlm.checkpoint import load_tensor_checkpoint


@dataclass(frozen=True)
class PrivacyAuditThresholds:
    min_reconstruction_nrmse: float = 0.90
    max_patient_linkage_rate: float = 0.25
    max_attribute_auc: float = 0.60
    max_membership_auc: float = 0.60
    min_samples: int = 100
    min_patients: int = 20

    def __post_init__(self) -> None:
        rates = (
            self.max_patient_linkage_rate,
            self.max_attribute_auc,
            self.max_membership_auc,
        )
        if not all(0 <= value <= 1 for value in rates):
            raise ValueError("privacy audit rate thresholds must lie in [0,1]")
        if self.min_reconstruction_nrmse < 0:
            raise ValueError("minimum reconstruction NRMSE must be non-negative")
        counts = (self.min_samples, self.min_patients)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in counts
        ):
            raise ValueError("privacy audit sample and patient minima must be integers")
        if self.min_samples < 4 or self.min_patients < 2:
            raise ValueError("privacy audit sample and patient minima are too small")

    def validate_promotion_policy(self) -> None:
        """Reject thresholds weaker than the repository's promotion policy."""
        policy = PrivacyAuditThresholds()
        weaker = (
            self.min_reconstruction_nrmse < policy.min_reconstruction_nrmse
            or self.max_patient_linkage_rate > policy.max_patient_linkage_rate
            or self.max_attribute_auc > policy.max_attribute_auc
            or self.max_membership_auc > policy.max_membership_auc
            or self.min_samples < policy.min_samples
            or self.min_patients < policy.min_patients
        )
        if weaker:
            raise ValueError("privacy audit thresholds are weaker than promotion policy")


@dataclass(frozen=True)
class RepresentationPrivacyAudit:
    schema_version: int
    patient_level: bool
    sample_count: int
    patient_count: int
    reconstruction_nrmse: float
    patient_linkage_rate: float
    attribute_auc: float
    membership_auc: float
    thresholds: PrivacyAuditThresholds
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["thresholds"] = asdict(self.thresholds)
        return value

    def save_json(self, path: str) -> None:
        resolved = os.path.abspath(os.fspath(path))
        with open(resolved, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


def _flatten_finite(name: str, value: torch.Tensor) -> np.ndarray:
    tensor = torch.as_tensor(value).detach().cpu().float()
    if tensor.ndim < 2 or tensor.shape[0] < 1:
        raise ValueError(f"{name} must have shape [samples, ...]")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor.reshape(tensor.shape[0], -1).numpy()


def _binary_or_macro_auc(target: np.ndarray, score: np.ndarray) -> float:
    classes = np.unique(target)
    if len(classes) < 2:
        raise ValueError("AUC attack target must contain at least two classes")
    if len(classes) == 2:
        positive = score[:, 1] if score.ndim == 2 else score
        auc = float(roc_auc_score(target, positive))
        return max(auc, 1.0 - auc)
    encoded = label_binarize(target, classes=classes)
    return float(roc_auc_score(encoded, score, average="macro", multi_class="ovr"))


def evaluate_finding_conditional(
    releases: torch.Tensor,
    sensitive_attributes: torch.Tensor,
    membership_labels: torch.Tensor,
    finding_families: list[str],
    *,
    seed: int = 42,
    min_family_samples: int = 8,
) -> Dict[str, Any]:
    """Attack AUC *within* finding families.

    A diagnostic bottleneck is supposed to preserve pathology. Unconditional
    membership labels that are stratified by finding family therefore mix
    diagnostic utility into the membership number. This report is informational:
    promotion still uses the unconditional gates.
    """
    x = _flatten_finite("releases", releases)
    attribute = torch.as_tensor(sensitive_attributes).detach().cpu().reshape(-1).numpy()
    membership = torch.as_tensor(membership_labels).detach().cpu().reshape(-1).numpy()
    if len(finding_families) != x.shape[0]:
        raise ValueError("finding_families must have one entry per release")
    families: Dict[str, Any] = {}
    sex_scores = []
    member_scores = []
    for name in sorted(set(finding_families)):
        idx = np.array([i for i, fam in enumerate(finding_families) if fam == name])
        if idx.size < min_family_samples:
            families[name] = {"skipped": "too_few_samples", "n": int(idx.size)}
            continue
        family_x = x[idx]
        family_sex = attribute[idx]
        family_mem = membership[idx]
        row: Dict[str, Any] = {"n": int(idx.size)}
        if len(np.unique(family_sex)) >= 2:
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(family_x, family_sex)
            auc = _binary_or_macro_auc(family_sex, model.predict_proba(family_x))
            row["attribute_auc"] = auc
            sex_scores.append(auc)
        else:
            row["attribute_auc"] = None
        if len(np.unique(family_mem)) >= 2:
            model = LogisticRegression(max_iter=1000, random_state=seed)
            model.fit(family_x, family_mem)
            auc = _binary_or_macro_auc(family_mem, model.predict_proba(family_x))
            row["membership_auc"] = auc
            member_scores.append(auc)
        else:
            row["membership_auc"] = None
        families[name] = row
    return {
        "schema_version": 1,
        "in_sample": True,
        "interpretation": (
            "Finding-conditional AUCs isolate residual leakage after pathology "
            "is held fixed. They do not replace the unconditional promotion gates."
        ),
        "macro_attribute_auc": (
            float(np.mean(sex_scores)) if sex_scores else None
        ),
        "macro_membership_auc": (
            float(np.mean(member_scores)) if member_scores else None
        ),
        "n_families_scored_attribute": len(sex_scores),
        "n_families_scored_membership": len(member_scores),
        "families": families,
    }


class PatientLevelPrivacyAuditor:
    """Fit deterministic white-box attacks against a released representation."""

    def __init__(
        self,
        thresholds: Optional[PrivacyAuditThresholds] = None,
        *,
        seed: int = 42,
    ):
        self.thresholds = thresholds or PrivacyAuditThresholds()
        self.thresholds.validate_promotion_policy()
        self.seed = int(seed)

    def evaluate(
        self,
        releases: torch.Tensor,
        reconstruction_targets: torch.Tensor,
        patient_ids: torch.Tensor,
        sensitive_attributes: torch.Tensor,
        membership_labels: torch.Tensor,
    ) -> RepresentationPrivacyAudit:
        x = _flatten_finite("releases", releases)
        y_recon = _flatten_finite("reconstruction_targets", reconstruction_targets)
        patient = torch.as_tensor(patient_ids).detach().cpu().reshape(-1).numpy()
        attribute = torch.as_tensor(sensitive_attributes).detach().cpu().reshape(-1).numpy()
        membership = torch.as_tensor(membership_labels).detach().cpu().reshape(-1).numpy()
        n = x.shape[0]
        if any(len(value) != n for value in (y_recon, patient, attribute, membership)):
            raise ValueError("all privacy audit inputs must have the same sample count")
        unique_patients, patient_counts = np.unique(patient, return_counts=True)
        if n < self.thresholds.min_samples:
            raise ValueError(
                f"privacy audit requires at least {self.thresholds.min_samples} samples"
            )
        if len(unique_patients) < self.thresholds.min_patients:
            raise ValueError(
                f"privacy audit requires at least {self.thresholds.min_patients} patients"
            )
        if np.any(patient_counts < 2):
            raise ValueError("patient linkage requires at least two studies per patient")

        attack_splits = self._select_attack_splits(
            x, patient, attribute, membership
        )
        reconstructed = np.zeros_like(y_recon)
        reconstruction_baseline = np.zeros_like(y_recon)
        attribute_score = np.zeros((n, len(np.unique(attribute))), dtype=np.float64)
        membership_score = np.zeros((n, len(np.unique(membership))), dtype=np.float64)
        for train_idx, test_idx in attack_splits:
            reconstructed[test_idx] = Ridge(alpha=1.0).fit(
                x[train_idx], y_recon[train_idx]
            ).predict(x[test_idx])
            reconstruction_baseline[test_idx] = np.mean(
                y_recon[train_idx], axis=0, keepdims=True
            )
            attr_model = LogisticRegression(max_iter=1000, random_state=self.seed)
            attr_model.fit(x[train_idx], attribute[train_idx])
            attribute_score[test_idx] = attr_model.predict_proba(x[test_idx])
            member_model = LogisticRegression(max_iter=1000, random_state=self.seed)
            member_model.fit(x[train_idx], membership[train_idx])
            membership_score[test_idx] = member_model.predict_proba(x[test_idx])
        rmse = float(np.sqrt(np.mean((reconstructed - y_recon) ** 2)))
        baseline_rmse = float(
            np.sqrt(np.mean((reconstruction_baseline - y_recon) ** 2))
        )
        reconstruction_nrmse = rmse / max(baseline_rmse, 1e-12)

        enrollment = []
        queries = []
        query_labels = []
        for label in unique_patients:
            indices = np.flatnonzero(patient == label)
            enrollment.append(x[indices[0]])
            queries.extend(x[index] for index in indices[1:])
            query_labels.extend([label] * (len(indices) - 1))
        enrollment_array = np.asarray(enrollment)
        query_array = np.asarray(queries)
        enrollment_array /= np.maximum(
            np.linalg.norm(enrollment_array, axis=1, keepdims=True), 1e-12
        )
        query_array /= np.maximum(
            np.linalg.norm(query_array, axis=1, keepdims=True), 1e-12
        )
        predictions = unique_patients[np.argmax(query_array @ enrollment_array.T, axis=1)]
        patient_linkage_rate = float(np.mean(predictions == np.asarray(query_labels)))

        attribute_auc = _binary_or_macro_auc(attribute, attribute_score)
        membership_auc = _binary_or_macro_auc(membership, membership_score)

        thresholds = self.thresholds
        passed = (
            reconstruction_nrmse >= thresholds.min_reconstruction_nrmse
            and patient_linkage_rate <= thresholds.max_patient_linkage_rate
            and attribute_auc <= thresholds.max_attribute_auc
            and membership_auc <= thresholds.max_membership_auc
        )
        return RepresentationPrivacyAudit(
            schema_version=1,
            patient_level=True,
            sample_count=n,
            patient_count=len(unique_patients),
            reconstruction_nrmse=reconstruction_nrmse,
            patient_linkage_rate=patient_linkage_rate,
            attribute_auc=attribute_auc,
            membership_auc=membership_auc,
            thresholds=thresholds,
            passed=passed,
        )

    def _select_attack_splits(
        self,
        releases: np.ndarray,
        patient_ids: np.ndarray,
        sensitive_attributes: np.ndarray,
        membership_labels: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Build patient-disjoint out-of-fold attacks with joint class balance."""
        targets = {
            "sensitive_attributes": sensitive_attributes,
            "membership_labels": membership_labels,
        }
        expected_classes = {}
        for name, target in targets.items():
            classes = np.unique(target)
            if len(classes) < 2:
                raise ValueError(f"{name} must contain at least two classes")
            expected_classes[name] = classes

        patient_targets = {}
        for patient in np.unique(patient_ids):
            indices = patient_ids == patient
            values = tuple(np.unique(target[indices]) for target in targets.values())
            if any(len(value) != 1 for value in values):
                raise ValueError("attack labels must be constant within each patient")
            patient_targets[patient] = tuple(int(value[0]) for value in values)
        strata = {value: index for index, value in enumerate(sorted(set(patient_targets.values())))}
        patient_strata = np.asarray(
            [strata[patient_targets[value]] for value in patient_targets]
        )
        minimum = min(np.sum(patient_strata == value) for value in np.unique(patient_strata))
        n_splits = min(5, int(minimum))
        if n_splits < 2:
            raise ValueError(
                "no patient-disjoint split covers every sensitive-attribute and "
                "membership class in both train and test sets"
            )
        generator = np.random.default_rng(self.seed)
        fold_patients = [[] for _ in range(n_splits)]
        patient_values = np.asarray(list(patient_targets))
        for value in np.unique(patient_strata):
            ordered = generator.permutation(patient_values[patient_strata == value])
            for index, patient in enumerate(ordered):
                fold_patients[index % n_splits].append(patient)
        splits = []
        for test_patients in fold_patients:
            test = np.flatnonzero(np.isin(patient_ids, test_patients))
            train = np.flatnonzero(~np.isin(patient_ids, test_patients))
            splits.append((train, test))
        if not all(
            all(
                np.array_equal(np.unique(target[train]), expected_classes[name])
                and np.array_equal(np.unique(target[test]), expected_classes[name])
                for name, target in targets.items()
            )
            for train, test in splits
        ):
            raise ValueError(
                "no patient-disjoint split covers every sensitive-attribute and "
                "membership class in both train and test sets"
            )
        return splits


def validate_privacy_audit(
    metrics: Mapping[str, Any],
    *,
    require_passed: bool = True,
) -> Mapping[str, Any]:
    """Validate the patient-level audit embedded in an FDBR calibration."""
    audit = metrics.get("privacy_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("secure FDBR calibration requires a privacy_audit mapping")
    required = {
        "schema_version",
        "patient_level",
        "sample_count",
        "patient_count",
        "reconstruction_nrmse",
        "patient_linkage_rate",
        "attribute_auc",
        "membership_auc",
        "thresholds",
        "passed",
    }
    missing = sorted(required - set(audit))
    if missing:
        raise ValueError(f"privacy_audit is missing fields: {missing}")
    if (
        not isinstance(audit["schema_version"], int)
        or isinstance(audit["schema_version"], bool)
        or audit["schema_version"] != 1
        or audit["patient_level"] is not True
    ):
        raise ValueError("privacy_audit must use schema 1 and patient-level identities")
    if not isinstance(audit["passed"], bool):
        raise ValueError("privacy_audit passed must be a boolean")
    numeric = (
        "reconstruction_nrmse",
        "patient_linkage_rate",
        "attribute_auc",
        "membership_auc",
    )
    if not all(np.isfinite(float(audit[key])) for key in numeric):
        raise ValueError("privacy_audit contains non-finite metrics")
    rates = (
        float(audit["patient_linkage_rate"]),
        float(audit["attribute_auc"]),
        float(audit["membership_auc"]),
    )
    if float(audit["reconstruction_nrmse"]) < 0 or not all(
        0 <= value <= 1 for value in rates
    ):
        raise ValueError("privacy_audit metrics are outside their valid ranges")
    counts = (audit["sample_count"], audit["patient_count"])
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
        for value in counts
    ):
        raise ValueError("privacy_audit sample and patient counts must be positive integers")
    thresholds = audit["thresholds"]
    if not isinstance(thresholds, Mapping):
        raise ValueError("privacy_audit thresholds must be a mapping")
    threshold_keys = {
        "min_reconstruction_nrmse",
        "max_patient_linkage_rate",
        "max_attribute_auc",
        "max_membership_auc",
        "min_samples",
        "min_patients",
    }
    missing_thresholds = sorted(threshold_keys - set(thresholds))
    if missing_thresholds:
        raise ValueError(
            f"privacy_audit thresholds are missing fields: {missing_thresholds}"
        )
    declared = PrivacyAuditThresholds(**{
        key: thresholds[key] for key in threshold_keys
    })
    declared.validate_promotion_policy()
    recomputed_pass = (
        int(audit["sample_count"]) >= int(thresholds["min_samples"])
        and int(audit["patient_count"]) >= int(thresholds["min_patients"])
        and float(audit["reconstruction_nrmse"])
        >= float(thresholds["min_reconstruction_nrmse"])
        and float(audit["patient_linkage_rate"])
        <= float(thresholds["max_patient_linkage_rate"])
        and float(audit["attribute_auc"]) <= float(thresholds["max_attribute_auc"])
        and float(audit["membership_auc"])
        <= float(thresholds["max_membership_auc"])
    )
    if bool(audit["passed"]) != recomputed_pass:
        raise ValueError("privacy_audit passed flag is inconsistent with its metrics")
    if require_passed and not recomputed_pass:
        raise ValueError("patient-level privacy audit did not pass all promotion gates")
    return audit


def validate_release_calibration(
    metrics: Mapping[str, Any],
    *,
    require_privacy_audit: bool,
    expected_mechanism: Optional[Mapping[str, Any]] = None,
    allow_infeasible: bool = False,
) -> None:
    """Enforce the utility/linkage and attack-audit release-promotion gates."""
    feasible = metrics.get("selected_candidate_feasible")
    if require_privacy_audit:
        if feasible is not True:
            raise ValueError(
                "secure release requires selected_candidate_feasible=true"
            )
        validate_privacy_audit(metrics)
        evidence = metrics.get("evidence_scope")
        if not isinstance(evidence, Mapping):
            raise ValueError("secure release calibration is missing evidence_scope")
        required_evidence = {
            "status",
            "external_confirmation",
            "attack_protocol",
            "attack_sample_count",
            "attack_patient_count",
            "privacy_cohort_used_for_adversarial_training",
        }
        missing_evidence = sorted(required_evidence - set(evidence))
        if missing_evidence:
            raise ValueError(
                f"secure release evidence_scope is missing fields: {missing_evidence}"
            )
        if evidence["status"] != "external_confirmation":
            raise ValueError(
                "secure release requires independent external confirmation; "
                "development feasibility is research-only"
            )
        if not isinstance(evidence["external_confirmation"], bool) or not isinstance(
            evidence["privacy_cohort_used_for_adversarial_training"], bool
        ):
            raise ValueError("secure release evidence_scope flags must be boolean")
        if evidence["attack_protocol"] != "five_fold_patient_disjoint_out_of_fold":
            raise ValueError("secure release requires the patient-disjoint OOF attack protocol")
        audit = metrics["privacy_audit"]
        if (
            evidence["attack_sample_count"] != audit["sample_count"]
            or evidence["attack_patient_count"] != audit["patient_count"]
        ):
            raise ValueError("secure release evidence_scope counts do not match privacy_audit")
        if evidence["external_confirmation"] is not True:
            raise ValueError("secure release requires external_confirmation=true")
        if evidence["privacy_cohort_used_for_adversarial_training"] is not False:
            raise ValueError(
                "secure release confirmation cohort must be independent of adversarial training"
            )
    elif feasible is False and not allow_infeasible:
        raise ValueError(
            "FDBR calibration is marked infeasible and cannot be used for release"
        )
    if expected_mechanism is not None:
        mechanism = metrics.get("mechanism")
        if not isinstance(mechanism, Mapping):
            raise ValueError("secure release calibration is missing mechanism parameters")
        for key, expected in expected_mechanism.items():
            if key not in mechanism:
                raise ValueError(f"secure release mechanism is missing {key!r}")
            actual = mechanism[key]
            if isinstance(expected, float):
                matches = isinstance(actual, (int, float)) and math.isclose(
                    float(actual), expected, rel_tol=1e-9, abs_tol=1e-12
                )
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(
                    f"secure release mechanism {key!r} mismatch: "
                    f"audited={actual!r}, configured={expected!r}"
                )


def bind_privacy_audit_to_calibration(
    calibration_path: str,
    audit: RepresentationPrivacyAudit,
    output_path: str,
) -> None:
    """Write a new tensor-only FDBR calibration with an embedded attack audit."""
    validate_privacy_audit({"privacy_audit": audit.to_dict()})
    source = os.path.abspath(os.fspath(calibration_path))
    target = os.path.abspath(os.fspath(output_path))
    if source == target:
        raise ValueError("privacy audit binding requires a new output checkpoint")
    if os.path.exists(target):
        raise FileExistsError(f"refusing to overwrite existing calibration: {target}")
    data = load_tensor_checkpoint(
        source,
        required_keys=("projection", "subspace", "k", "label_classes", "metrics"),
    )
    metrics = dict(data["metrics"])
    if metrics.get("selected_candidate_feasible") is not True:
        raise ValueError("cannot promote a calibration whose utility/linkage gate failed")
    metrics["privacy_audit"] = audit.to_dict()
    promoted = dict(data)
    promoted["metrics"] = metrics
    torch.save(promoted, target)


__all__ = [
    "PatientLevelPrivacyAuditor",
    "PrivacyAuditThresholds",
    "RepresentationPrivacyAudit",
    "bind_privacy_audit_to_calibration",
    "evaluate_finding_conditional",
    "validate_privacy_audit",
    "validate_release_calibration",
]
