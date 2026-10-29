"""Schema and conversion helpers for callback-free FHE checkpoints.

The interactive CADN checkpoint is intentionally not loadable as a
``model_confidential`` checkpoint.  This module provides a deterministic,
auditable conversion for research distillation and a strict validation
contract for production loading.  Conversion is only an initialization step;
the resulting artifact must carry measured validation evidence before it can
be deployed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import torch

from .checkpoint import load_tensor_checkpoint

SCHEMA_VERSION = 1
CONTINUOUS_PROFILE = "model_confidential"
MIN_HOLDOUT_COSINE_MEAN = 0.98
MIN_HOLDOUT_COSINE_MIN = 0.95
MAX_HOLDOUT_RELATIVE_L2_P95 = 0.30


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_region_affine(
    weight: torch.Tensor,
    bias: torch.Tensor,
    ema_mean: torch.Tensor,
    ema_mean_x2: torch.Tensor,
    num_regions: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate dynamic LayerNorm with PolyNorm scalar region coefficients."""
    variance = (ema_mean_x2 - ema_mean.square()).clamp_min(1e-6)
    slope = weight / variance.sqrt()
    intercept = bias - slope * ema_mean
    c1 = slope.mean().repeat(num_regions)
    c0 = intercept.mean().repeat(num_regions)
    c2 = torch.zeros(num_regions, dtype=c1.dtype)
    return c0, c1, c2


def convert_cadn_checkpoint(
    source_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    num_layers: int,
    num_regions: int,
    continuous_depth: int = 21,
    preprocessing_fingerprint: Optional[str] = None,
    source_sha256: Optional[str] = None,
    architecture_variant: str = "canonical",
) -> dict[str, Any]:
    """Create a non-CADN initialization from a CADN checkpoint.

    The output is marked ``candidate`` and is rejected by secure loading until
    a benchmark promotes it to ``validated``.  No claim of utility retention
    is made by this conversion alone.
    """
    source = load_tensor_checkpoint(
        str(source_path), required_keys=("encoder_state_dict", "adapter_state_dict")
    )
    if source_sha256 is not None and sha256_file(source_path) != source_sha256.lower():
        raise ValueError("source checkpoint digest mismatch")
    enc = dict(source["encoder_state_dict"])
    converted: dict[str, torch.Tensor] = {}
    for key, value in enc.items():
        if ".norm1.layer_norm." in key:
            continue
        if ".norm1._ema_" in key:
            continue
        if key.startswith("final_norm.layer_norm."):
            continue
        converted[key] = value

    for index in range(num_layers):
        prefix = f"blocks.{index}.norm1"
        try:
            weight = enc[f"{prefix}.layer_norm.weight"]
            bias = enc[f"{prefix}.layer_norm.bias"]
            mean = enc[f"{prefix}._ema_mean"]
            mean_x2 = enc[f"{prefix}._ema_mean_x2"]
        except KeyError as exc:
            raise ValueError(
                f"source is not a CADN checkpoint; missing {exc.args[0]}"
            ) from exc
        c0, c1, c2 = _scalar_region_affine(
            weight, bias, mean, mean_x2, num_regions
        )
        converted[f"{prefix}.weight"] = weight
        converted[f"{prefix}.bias"] = bias
        if num_regions == 1:
            converted[f"{prefix}.c0"] = c0[:1].expand_as(weight)
            converted[f"{prefix}.c1"] = c1[:1].expand_as(weight)
            converted[f"{prefix}.c2"] = c2[:1].expand_as(weight)
        else:
            converted[f"{prefix}.c0_regions"] = c0
            converted[f"{prefix}.c1_regions"] = c1
            converted[f"{prefix}.c2_regions"] = c2

    for name in ("weight", "bias"):
        key = f"final_norm.layer_norm.{name}"
        if key not in enc:
            raise ValueError(f"source is not a CADN checkpoint; missing {key}")
        converted[f"final_norm.{name}"] = enc[key]
    for name in ("_ema_mean", "_ema_mean_x2"):
        key = f"final_norm.{name}"
        if key not in enc:
            raise ValueError(f"source is not a CADN checkpoint; missing {key}")
        converted[key] = enc[key]

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "profile": CONTINUOUS_PROFILE,
        "checkpoint_kind": "continuous_fhe",
        "conversion": "cadn_to_polynorm_affine_initialization",
        "source_sha256": sha256_file(source_path),
        "architecture": {
            "num_layers": num_layers,
            "num_regions": num_regions,
            "use_cadn": False,
            "continuous_depth": continuous_depth,
            "poly_norm_degree": 1,
            "linear_polynorm_fusion": False,
            "fuse_linear_poly_norm": False,
            "channelwise_polynorm": False,
            "architecture_variant": architecture_variant,
        },
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "validation": {"status": "candidate", "evidence": None},
    }
    payload = {
        "encoder_state_dict": converted,
        "adapter_state_dict": dict(source["adapter_state_dict"]),
        "metadata": metadata,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    # Write metadata next to the binary so promotion is reviewable without
    # unpickling the checkpoint.
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def validate_continuous_checkpoint(
    value: Mapping[str, Any],
    *,
    require_validated: bool = True,
    expected_depth: Optional[int] = None,
    expected_preprocessing_fingerprint: Optional[str] = None,
    expected_linear_polynorm_fusion: Optional[bool] = None,
    expected_channelwise_polynorm: Optional[bool] = None,
    expected_architecture_variant: Optional[str] = None,
) -> Mapping[str, Any]:
    """Validate the metadata contract before constructing a confidential model."""
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "model_confidential requires checkpoint metadata; refusing a legacy checkpoint"
        )
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported continuous checkpoint schema")
    if metadata.get("profile") != CONTINUOUS_PROFILE:
        raise ValueError("checkpoint profile is not model_confidential")
    if metadata.get("checkpoint_kind") != "continuous_fhe":
        raise ValueError("checkpoint is not a continuous-FHE artifact")
    architecture = metadata.get("architecture")
    if not isinstance(architecture, Mapping) or architecture.get("use_cadn") is not False:
        raise ValueError("continuous checkpoint architecture contract is missing or invalid")
    if expected_depth is not None and architecture.get("continuous_depth") != expected_depth:
        raise ValueError("continuous checkpoint depth does not match the configured circuit")
    if (
        expected_architecture_variant not in (None, "canonical")
        and architecture.get("architecture_variant") != expected_architecture_variant
    ):
        raise ValueError("continuous checkpoint architecture variant does not match the configured circuit")
    if (
        expected_linear_polynorm_fusion is True
        and not (
            architecture.get("linear_polynorm_fusion") is True
            or architecture.get("fuse_linear_poly_norm") is True
        )
    ):
        raise ValueError(
            "continuous checkpoint is not marked as linear-PolyNorm-fusion ready"
        )
    if (
        expected_channelwise_polynorm is True
        and architecture.get("channelwise_polynorm") is not True
    ):
        raise ValueError(
            "continuous checkpoint is not marked as channelwise-PolyNorm-ready"
        )
    if (
        expected_preprocessing_fingerprint is not None
        and metadata.get("preprocessing_fingerprint")
        != expected_preprocessing_fingerprint
    ):
        raise ValueError("continuous checkpoint preprocessing fingerprint mismatch")
    validation = metadata.get("validation")
    status = validation.get("status") if isinstance(validation, Mapping) else None
    if require_validated and status != "validated":
        raise ValueError(
            "model_confidential checkpoint is not validated; run the continuous "
            "benchmark and promote its evidence before deployment"
        )
    if status == "validated":
        metrics = validation.get("metrics") if isinstance(validation, Mapping) else None
        if not isinstance(metrics, Mapping):
            raise ValueError("validated continuous checkpoint is missing metrics")
        if int(validation.get("holdout_count", 0)) < 10:
            raise ValueError("validated continuous checkpoint requires at least 10 holdout images")
        if (
            float(metrics.get("cosine_mean", 0.0)) < MIN_HOLDOUT_COSINE_MEAN
            or float(metrics.get("cosine_min", 0.0)) < MIN_HOLDOUT_COSINE_MIN
            or float(metrics.get("relative_l2_p95", float("inf")))
            > MAX_HOLDOUT_RELATIVE_L2_P95
        ):
            raise ValueError("continuous checkpoint validation metrics fail the fixed gate")
    return metadata


__all__ = [
    "CONTINUOUS_PROFILE",
    "MAX_HOLDOUT_RELATIVE_L2_P95",
    "MIN_HOLDOUT_COSINE_MEAN",
    "MIN_HOLDOUT_COSINE_MIN",
    "SCHEMA_VERSION",
    "convert_cadn_checkpoint",
    "sha256_file",
    "validate_continuous_checkpoint",
]
