import hashlib
import tempfile
import unittest
from pathlib import Path

from experiments.common import diagnosis_target, select_member_patients
from experiments.download_patient_cohort import (
    ALLOWED_LICENSES,
    COMMIT,
    MIN_IMAGES_PER_PATIENT,
    _select_cohorts,
    validate_disjoint_cohorts,
    validate_manifest,
)


def _row(patient: str, sex: str, study: int, *, license_name: str = "CC BY 4.0"):
    return {
        "modality": "X-ray",
        "view": "PA",
        "patientid": patient,
        "filename": f"{patient}-{study}.png",
        "sex": sex,
        "license": license_name,
        "date": f"2020-01-{study + 1:02d}",
        "finding": "Pneumonia",
    }


class PatientCohortTests(unittest.TestCase):
    def test_selection_is_deterministic_balanced_and_license_filtered(self):
        rows = []
        for sex in ("F", "M"):
            for patient_index in range(6):
                patient = f"{sex}-{patient_index:02d}"
                rows.extend(_row(patient, sex, study) for study in range(4))
        rows.extend(_row("blocked", "F", study, license_name="CC BY-NC-ND") for study in range(3))

        first = _select_cohorts(rows)
        second = _select_cohorts(list(reversed(rows)))
        self.assertEqual(first, second)
        development = {record["patient_id"] for record in first["development"]}
        confirmation = {record["patient_id"] for record in first["confirmation"]}
        self.assertTrue(development.isdisjoint(confirmation))
        self.assertEqual(len(development | confirmation), 12)
        self.assertNotIn("blocked", development | confirmation)
        for sex in ("F", "M"):
            development_sex = {
                record["patient_id"] for record in first["development"] if record["sex"] == sex
            }
            confirmation_sex = {
                record["patient_id"] for record in first["confirmation"] if record["sex"] == sex
            }
            self.assertEqual(len(development_sex), 3)
            self.assertEqual(len(confirmation_sex), 3)
        all_records = first["development"] + first["confirmation"]
        self.assertEqual(len(all_records), 48)
        self.assertTrue(all(record["license"] in ALLOWED_LICENSES for record in all_records))
        validate_disjoint_cohorts(
            {"records": first["development"]},
            {"records": first["confirmation"]},
        )

        overlapping = {"records": [dict(first["development"][0])]}
        with self.assertRaisesRegex(ValueError, "share patient"):
            validate_disjoint_cohorts(overlapping, overlapping)

    def test_exactly_two_study_patients_are_reserved_for_final_holdout(self):
        rows = []
        for sex in ("F", "M"):
            for patient_index in range(4):
                rows.extend(
                    _row(f"train-{sex}-{patient_index}", sex, study)
                    for study in range(MIN_IMAGES_PER_PATIENT)
                )
            for patient_index in range(3):
                rows.extend(
                    _row(f"holdout-{sex}-{patient_index}", sex, study)
                    for study in range(2)
                )
        cohorts = _select_cohorts(rows)
        final_patients = {row["patient_id"] for row in cohorts["final_holdout"]}
        earlier_patients = {
            row["patient_id"]
            for split in ("development", "confirmation")
            for row in cohorts[split]
        }
        self.assertEqual(len(final_patients), 6)
        self.assertTrue(final_patients.isdisjoint(earlier_patients))
        self.assertTrue(all(name.startswith("holdout-") for name in final_patients))

    def test_selection_rejects_insufficient_repeated_patients(self):
        rows = []
        for sex in ("F", "M"):
            patient = f"{sex}-00"
            rows.extend(_row(patient, sex, study) for study in range(MIN_IMAGES_PER_PATIENT))
        with self.assertRaisesRegex(ValueError, "eligible repeated patients"):
            _select_cohorts(rows)

    def test_membership_is_deterministic_balanced_and_labels_are_honest(self):
        records = []
        findings = (
            ["todo"] * 3
            + ["Pneumonia/Viral/COVID-19"] * 7
            + ["Pneumonia/Bacterial/Streptococcus"] * 5
            + ["Pneumonia"] * 5
        )
        for sex in ("F", "M"):
            for index, finding in enumerate(findings):
                records.append(
                    {
                        "patient_id": f"{sex}-{index:02d}",
                        "sex": sex,
                        "finding": finding,
                    }
                )
        first = select_member_patients(records)
        second = select_member_patients(list(reversed(records)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(sum(patient.startswith("F-") for patient in first), 10)
        self.assertEqual(sum(patient.startswith("M-") for patient in first), 10)
        self.assertEqual(diagnosis_target("todo"), ([0.0] * 4, False))
        self.assertEqual(
            diagnosis_target("Pneumonia/Viral/COVID-19"),
            ([1.0, 1.0, 1.0, 0.0], True),
        )

    def test_manifest_validation_checks_digests_and_patient_design(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            records = []
            for sex in ("F", "M"):
                for patient_index in range(20):
                    patient = f"{sex}-{patient_index:02d}"
                    for study in range(MIN_IMAGES_PER_PATIENT):
                        relative = Path("images") / f"{patient}-{study}.png"
                        content = f"{patient}:{study}".encode()
                        (root / relative).write_bytes(content)
                        records.append(
                            {
                                "patient_id": patient,
                                "study_index": study,
                                "sex": sex,
                                "license": "CC BY 4.0",
                                "relative_path": relative.as_posix(),
                                "sha256": hashlib.sha256(content).hexdigest(),
                            }
                        )
            manifest = {
                "schema_version": 1,
                "commit": COMMIT,
                "selection": {
                    "minimum_images_per_patient": MIN_IMAGES_PER_PATIENT,
                    "image_count": len(records),
                    "patient_count": 40,
                    "patients_by_sex": {"F": 20, "M": 20},
                },
                "records": records,
            }
            validate_manifest(manifest, root)
            records[0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                validate_manifest(manifest, root)


if __name__ == "__main__":
    unittest.main()
