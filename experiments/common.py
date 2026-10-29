"""Shared, integrity-checked utilities for the maintained experiments."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.download_patient_cohort import validate_manifest  # noqa: E402
from fpavlm.checkpoint import load_artifact_manifest, load_tensor_checkpoint  # noqa: E402
from fpavlm.preprocessing import preprocess_image as canonical_preprocess_image  # noqa: E402

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
ARTIFACT_MANIFEST = PROJECT_ROOT / "artifacts.manifest.json"
ENCODER_WEIGHTS = ARTIFACT_ROOT / "encoder_weights.pt"
MEDICAL_PRIORS = ARTIFACT_ROOT / "medical_priors.pt"
FDBR_CALIBRATION = ARTIFACT_ROOT / "fdbr_calibration_development.pt"
OPENI_ROOT = PROJECT_ROOT / "data" / "openi_subset"
OPENI_IMAGES_DIR = OPENI_ROOT / "images"
OPENI_METADATA = OPENI_ROOT / "metadata.json"
PATIENT_ROOT = PROJECT_ROOT / "data" / "covid_cxr2_privacy_development"
PATIENT_MANIFEST = PATIENT_ROOT / "metadata.json"
CONFIRMATION_ROOT = PROJECT_ROOT / "data" / "covid_cxr2_privacy_confirmation"
CONFIRMATION_MANIFEST = CONFIRMATION_ROOT / "metadata.json"
FINAL_HOLDOUT_ROOT = PROJECT_ROOT / "data" / "covid_cxr2_privacy_final_holdout"
FINAL_HOLDOUT_MANIFEST = FINAL_HOLDOUT_ROOT / "metadata.json"

LABEL_CLASSES = (
    "normal",
    "opacity",
    "calcified granuloma",
    "pulmonary atelectasis",
    "cardiomegaly",
)


def artifact_sha256(path: Path) -> str:
    manifest = load_artifact_manifest(str(ARTIFACT_MANIFEST))
    canonical = str(path.resolve()).lower()
    normalized = {key.lower(): value for key, value in manifest.items()}
    if canonical not in normalized:
        raise ValueError(f"artifact is absent from the integrity manifest: {path}")
    return normalized[canonical]


def preprocess_image(
    image_path: Union[str, Path], image_size: int = 224, input_range: float = 3.0
) -> torch.Tensor:
    return canonical_preprocess_image(
        image_path,
        image_size=image_size,
        input_range=input_range,
    )


def load_openi_records(limit: Optional[int] = None) -> list[tuple[dict, str]]:
    value = json.loads(OPENI_METADATA.read_text(encoding="utf-8"))
    records = value["value"] if isinstance(value, dict) and "value" in value else value
    matched = []
    for record in records:
        filename = record.get("image_filename")
        if not filename:
            raise ValueError("OpenI metadata record is missing image_filename")
        path = OPENI_IMAGES_DIR / filename
        if path.is_file():
            matched.append((record, str(path)))
        if limit is not None and len(matched) >= limit:
            break
    if not matched:
        raise ValueError("no OpenI metadata records have matching image files")
    return matched


def diagnosis_targets(records: list[dict]) -> np.ndarray:
    values = np.zeros((len(records), len(LABEL_CLASSES)), dtype=np.float32)
    for row, record in enumerate(records):
        problems = record.get("problems", "").strip().lower()
        labels = (
            {"normal"}
            if not problems or problems == "normal"
            else {item.strip() for item in problems.split(";") if item.strip()}
        )
        for column, name in enumerate(LABEL_CLASSES):
            values[row, column] = float(name in labels)
    return values


def build_encoder(
    encoder_weights: Path = ENCODER_WEIGHTS,
    *,
    expected_encoder_sha256: Optional[str] = None,
    use_cadn: Optional[bool] = None,
    poly_norm_degree: Optional[int] = None,
    poly_act_degree_schedule: Optional[list[int]] = None,
    fuse_linear_poly_act: Optional[bool] = None,
    fuse_linear_poly_norm: Optional[bool] = None,
    channelwise_polynorm: Optional[bool] = None,
    fuse_final_affine: Optional[bool] = None,
    enable_token_mix: Optional[bool] = None,
):
    from fpavlm import FHEConfig
    from fpavlm.cipher_native.anatomy_router import AnatomyAwareTokenRouter
    from fpavlm.cipher_native.noise_resilient_encoder import NoiseResilientMAViTEncoder

    config = FHEConfig(release_mode="legacy_embedding", dsr_calibration_path=None)
    if use_cadn is not None:
        config.use_cadn = bool(use_cadn)
    if poly_norm_degree is not None:
        config.poly_norm_degree = int(poly_norm_degree)
    if poly_act_degree_schedule is not None:
        if len(poly_act_degree_schedule) != config.num_layers:
            raise ValueError(
                "poly_act_degree_schedule length must match configured num_layers"
            )
        config.poly_act_degree_schedule = [int(value) for value in poly_act_degree_schedule]
    if fuse_linear_poly_act is not None:
        config.fuse_linear_poly_act = bool(fuse_linear_poly_act)
    if fuse_linear_poly_norm is not None:
        config.fuse_linear_poly_norm = bool(fuse_linear_poly_norm)
    if channelwise_polynorm is not None:
        config.channelwise_polynorm = bool(channelwise_polynorm)
    if enable_token_mix is not None:
        config.enable_token_mix = bool(enable_token_mix)
    if fuse_final_affine is not None:
        config.fuse_final_affine = bool(fuse_final_affine)
    priors = load_tensor_checkpoint(
        str(MEDICAL_PRIORS),
        expected_sha256=artifact_sha256(MEDICAL_PRIORS),
    )
    router = AnatomyAwareTokenRouter(
        dim=1,
        num_tokens=config.num_patches,
        grid_size=config.grid_size,
        num_regions=config.num_regions,
    )
    encoder = NoiseResilientMAViTEncoder(
        dim=config.dim,
        num_layers=config.num_layers,
        mlp_ratio=config.mlp_ratio,
        num_patches=config.num_patches,
        in_channels=config.in_channels,
        patch_size=config.patch_size,
        img_size=config.image_size,
        spectral_bound=config.spectral_bound,
        input_range=config.input_range,
        block_type="med_fhe_vit_v3",
        enable_noise_training=False,
        poly_act_degree_schedule=config.poly_act_degree_schedule,
        poly_norm_degree=config.poly_norm_degree,
        num_regions=config.num_regions,
        llm_dim=config.llm_hidden_dim,
        num_concepts=config.num_concepts,
        pool_type="perceiver_decoder",
        student_grid=config.student_grid,
        target_grid=config.target_grid,
        perceiver_config=config.perceiver_config,
        enable_token_mix=config.enable_token_mix,
        token_mix_spectral_bound=config.token_mix_spectral_bound,
        fuse_linear_poly_act=config.fuse_linear_poly_act,
        fuse_linear_poly_norm=config.fuse_linear_poly_norm,
        channelwise_polynorm=config.channelwise_polynorm,
        fuse_final_affine=config.fuse_final_affine,
        use_cadn=config.use_cadn,
        perceiver_name=config.perceiver_name,
        medical_prior=priors["medical_prior"],
        anatomical_prior=priors["anatomical_prior"],
        disease_prior_embeddings=priors["disease_prior_embeddings"],
        med_poly_act_prior=priors["med_poly_act_prior"],
        symmetry_pairs=router.get_symmetry_pairs(),
        disease_anchor_directions=priors["disease_anchor_directions"],
        disease_anchor_alpha=0.3,
        region_salience=priors["region_salience"],
    )
    checkpoint = load_tensor_checkpoint(
        str(encoder_weights),
        required_keys=("encoder_state_dict", "adapter_state_dict"),
        # Unregistered research candidates are accepted only with an explicit
        # caller-supplied digest.  The default remains manifest-pinned.
        expected_sha256=(
            expected_encoder_sha256
            if expected_encoder_sha256 is not None
            else artifact_sha256(encoder_weights)
        ),
    )
    state = dict(checkpoint["encoder_state_dict"])
    state.update(
        {
            f"concept_adapter.{name}": value
            for name, value in checkpoint["adapter_state_dict"].items()
        }
    )
    if config.channelwise_polynorm:
        # Bootstrap a channelwise student from legacy scalar-region weights by
        # broadcasting each region coefficient across feature channels. This
        # is an explicit architecture conversion; the resulting checkpoint is
        # still required to pass the holdout gate before deployment.
        for key in (
            "blocks.0.norm1.c0_regions", "blocks.0.norm1.c1_regions",
            "blocks.0.norm1.c2_regions", "blocks.1.norm1.c0_regions",
            "blocks.1.norm1.c1_regions", "blocks.1.norm1.c2_regions",
            "blocks.2.norm1.c0_regions", "blocks.2.norm1.c1_regions",
            "blocks.2.norm1.c2_regions",
        ):
            if key in state:
                target = encoder.state_dict().get(key)
                if target is not None and state[key].shape != target.shape:
                    if state[key].ndim == 1 and target.ndim == 2:
                        state[key] = state[key].unsqueeze(1).expand_as(target).clone()
    load_result = encoder.load_state_dict(state, strict=enable_token_mix is None)
    if enable_token_mix is not None and load_result.missing_keys:
        raise RuntimeError(
            "architecture override dropped required checkpoint keys: "
            f"{load_result.missing_keys}"
        )
    encoder.eval()
    encoder.bake_statistics()
    return encoder, config


def extract_vit_features(
    encoder,
    image_paths: list[str],
    *,
    device: Union[str, torch.device] = "cpu",
    input_range: float = 3.0,
) -> torch.Tensor:
    encoder = encoder.to(device)
    rows = []
    image_size = encoder.grid_size * encoder.patch_size
    for index, image_path in enumerate(image_paths, start=1):
        image = preprocess_image(
            image_path,
            image_size=image_size,
            input_range=input_range,
        ).to(device)
        with torch.no_grad():
            tokens = encoder.patch_proj(encoder.patchify(image))
            tokens = tokens + encoder.pos_embed[:, : encoder.num_patches, :]
            for block in encoder.blocks:
                tokens = block(tokens)
            tokens = encoder.final_norm(tokens)
        rows.append(tokens.squeeze(0).cpu())
        if index % 20 == 0:
            print(f"extracted {index}/{len(image_paths)} images", flush=True)
    return torch.stack(rows)


def _stable_key(value: str) -> str:
    return hashlib.sha256(f"FHEViT-patient-audit-v1:{value}".encode()).hexdigest()


def _finding_family(finding: str) -> str:
    value = finding.strip().lower()
    if value in {"", "todo", "unknown"}:
        return "unlabeled"
    if "covid-19" in value:
        return "covid_19"
    if "bacterial" in value:
        return "bacterial"
    if "viral" in value:
        return "other_viral"
    if "fungal" in value:
        return "fungal"
    if value == "no finding":
        return "no_finding"
    return "pneumonia_unspecified"


def diagnosis_target(finding: str) -> tuple[list[float], bool]:
    value = finding.strip().lower()
    if value in {"", "todo", "unknown"}:
        return [0.0] * 4, False
    return [
        float("pneumonia" in value),
        float("viral" in value),
        float("covid-19" in value),
        float("bacterial" in value),
    ], True


def select_member_patients(records: list[dict]) -> set[str]:
    by_patient = {record["patient_id"]: record for record in records}
    selected: set[str] = set()
    for sex in ("F", "M"):
        categories: dict[str, list[str]] = defaultdict(list)
        for patient, record in by_patient.items():
            if record["sex"] == sex:
                categories[_finding_family(record["finding"])].append(patient)
        sex_selected: set[str] = set()
        for category in sorted(categories):
            ordered = sorted(categories[category], key=_stable_key)
            sex_selected.update(ordered[: len(ordered) // 2])
        all_sex = sorted(
            (patient for patient, record in by_patient.items() if record["sex"] == sex),
            key=_stable_key,
        )
        target_count = len(all_sex) // 2
        for patient in all_sex:
            if len(sex_selected) >= target_count:
                break
            sex_selected.add(patient)
        if len(sex_selected) > target_count:
            sex_selected = set(sorted(sex_selected, key=_stable_key)[:target_count])
        if len(sex_selected) != target_count:
            raise ValueError(f"could not select balanced member patients for sex {sex}")
        selected.update(sex_selected)
    return selected


def load_patient_inputs(
    device: Union[str, torch.device] = "cpu",
    *,
    root: Path = PATIENT_ROOT,
):
    manifest_path = root / "metadata.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, root)
    records = manifest["records"]
    paths = [str(root / record["relative_path"]) for record in records]
    encoder, config = build_encoder()
    features = extract_vit_features(
        encoder, paths, device=device, input_range=config.input_range
    ).float()
    pixels = []
    for path in paths:
        with Image.open(path) as image:
            resized = image.convert("L").resize(
                (32, 32), Image.Resampling.BILINEAR
            )
            pixels.append(
                torch.from_numpy(np.asarray(resized, dtype=np.float32).copy() / 255.0)
            )
    patient_names = sorted({record["patient_id"] for record in records}, key=_stable_key)
    patient_index = {name: index for index, name in enumerate(patient_names)}
    patients = torch.tensor([patient_index[record["patient_id"]] for record in records])
    sex = torch.tensor([0 if record["sex"] == "F" else 1 for record in records])
    members = select_member_patients(records)
    membership = torch.tensor([int(record["patient_id"] in members) for record in records])
    return records, features, torch.stack(pixels), patients, sex, membership


def worst_case_audit(audits):
    from fpavlm.privacy_audit import RepresentationPrivacyAudit

    thresholds = audits[0].thresholds
    return RepresentationPrivacyAudit(
        schema_version=1,
        patient_level=True,
        sample_count=audits[0].sample_count,
        patient_count=audits[0].patient_count,
        reconstruction_nrmse=min(audit.reconstruction_nrmse for audit in audits),
        patient_linkage_rate=max(audit.patient_linkage_rate for audit in audits),
        attribute_auc=max(audit.attribute_auc for audit in audits),
        membership_auc=max(audit.membership_auc for audit in audits),
        thresholds=thresholds,
        passed=all(audit.passed for audit in audits),
    )


__all__ = [
    "ARTIFACT_MANIFEST",
    "ARTIFACT_ROOT",
    "CONFIRMATION_MANIFEST",
    "CONFIRMATION_ROOT",
    "FINAL_HOLDOUT_MANIFEST",
    "FINAL_HOLDOUT_ROOT",
    "ENCODER_WEIGHTS",
    "FDBR_CALIBRATION",
    "LABEL_CLASSES",
    "MEDICAL_PRIORS",
    "OPENI_IMAGES_DIR",
    "OPENI_METADATA",
    "PATIENT_MANIFEST",
    "PROJECT_ROOT",
    "artifact_sha256",
    "build_encoder",
    "diagnosis_target",
    "diagnosis_targets",
    "extract_vit_features",
    "load_openi_records",
    "load_patient_inputs",
    "preprocess_image",
    "select_member_patients",
    "worst_case_audit",
]
