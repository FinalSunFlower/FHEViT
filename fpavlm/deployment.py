"""Deployment contracts and conservative CKKS budget checks."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class DeploymentProfile(str, Enum):
    """Explicit trust and execution profiles supported by FPA-VLM."""

    RESEARCH_BASELINE = "research_baseline"
    UTILITY_INTERACTIVE = "utility_interactive"
    MODEL_CONFIDENTIAL = "model_confidential"


# Microsoft SEAL's documented tc128 maximum coefficient-modulus bit counts.
# This is a conservative configuration gate, not a substitute for reporting the
# exact library version and running a parameter estimator in an experiment.
TC128_MAX_COEFF_MODULUS_BITS = {8192: 218, 16384: 438, 32768: 881}


@dataclass(frozen=True)
class CKKSDepthBudget:
    multiplicative_depth: int
    coefficient_modulus_bits: int
    tc128_max_bits: int


def continuous_fhe_depth(
    polynomial_degrees: Sequence[int],
    *,
    token_mixing: bool,
    poly_norm_degree: int = 2,
    include_patch_projection: bool = True,
    include_final_affine: bool = True,
    include_release_projection: bool = True,
    fuse_linear_poly_act: bool = False,
    fuse_linear_poly_norm: bool = False,
    fuse_final_affine: bool = False,
) -> int:
    """Return the longest no-refresh path implemented by the encoder.

    Each non-CADN block contains degree-2 PolyNorm, one input projection, a
    degree-1/2 activation, one output projection, and optional token mixing.
    All plaintext multiplications consume a TenSEAL level with auto-rescale.
    """
    if not polynomial_degrees:
        raise ValueError("polynomial_degrees must not be empty")
    if any(degree not in (1, 2) for degree in polynomial_degrees):
        raise ValueError("only degree-1 and degree-2 activations are supported")
    if poly_norm_degree not in (1, 2):
        raise ValueError("poly_norm_degree must be 1 or 2")
    depth = int(include_patch_projection)
    depth += sum(
        poly_norm_degree
        + 1
        + (0 if fuse_linear_poly_act and degree == 1 else degree)
        + 1
        + int(token_mixing)
        for degree in polynomial_degrees
    )
    if fuse_linear_poly_norm and poly_norm_degree == 1:
        depth -= len(polynomial_degrees)
    depth += int(include_final_affine and not fuse_final_affine)
    depth += int(include_release_projection)
    return depth


def validate_ckks_budget(
    poly_modulus_degree: int,
    scale_bits: int,
    depth_budget: int,
) -> CKKSDepthBudget:
    """Reject unsupported or over-budget CKKS parameter combinations."""
    if poly_modulus_degree not in TC128_MAX_COEFF_MODULUS_BITS:
        raise ValueError(
            "poly_modulus_degree must be one of "
            f"{sorted(TC128_MAX_COEFF_MODULUS_BITS)}, got {poly_modulus_degree}"
        )
    if scale_bits < 20:
        raise ValueError("scale_bits below 20 is unsupported for this CKKS pipeline")
    if depth_budget < 1:
        raise ValueError("depth_budget must be positive")
    total_bits = 60 + scale_bits * depth_budget + 60
    maximum = TC128_MAX_COEFF_MODULUS_BITS[poly_modulus_degree]
    if total_bits > maximum:
        raise ValueError(
            f"CKKS coefficient modulus requires {total_bits} bits but the "
            f"tc128 limit for N={poly_modulus_degree} is {maximum} bits"
        )
    return CKKSDepthBudget(depth_budget, total_bits, maximum)


__all__ = [
    "CKKSDepthBudget",
    "DeploymentProfile",
    "TC128_MAX_COEFF_MODULUS_BITS",
    "continuous_fhe_depth",
    "validate_ckks_budget",
]
