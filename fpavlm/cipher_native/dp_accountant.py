"""Differential Privacy Accountant for CFP (v166 Innovation: Formal (ε, δ)-DP Guarantee).

EXTENDS v154/v164 CFP (Calibrated Feature Perturbation) with formal (ε, δ)-DP
guarantee via Rényi Differential Privacy (RDP) accounting. This closes the core
gap identified in the 2026-07-25 audit: CFP σ=24× achieved empirical 5% chance
level under linear attacks, but lacked formal privacy guarantee against
unbounded attackers.

FIRST-PRINCIPLES ANALYSIS:
  The Gaussian mechanism releases f(D) + N(0, σ²I_d) and satisfies (ε, δ)-DP when
  σ ≥ Δ₂f · √(2 ln(1.25/δ)) / ε, where Δ₂f is the L2 sensitivity of f.

  For embedding release:
    - f(image) = FHE_ViT(image) ∈ ℝ^d, d = 16 × 2560 = 40960
    - Without clipping, Δ₂f is unbounded → no DP guarantee
    - With L2 clipping to ||f||₂ ≤ R: Δ₂f = 2R (worst-case over neighboring images)

  RDP (Mironov 2017) gives tighter bounds than basic Gaussian:
    - Gaussian mechanism satisfies (α, α·Δ₂f²/(2σ²))-RDP for any α > 1
    - Convert to (ε, δ)-DP: ε = min_α [α·Δ₂f²/(2σ²) + ln(1/δ)/(α-1)]

AUDIT CORRECTION (v177):
  Earlier versions incorrectly substituted the expected L2 norm of a d-dimensional
  Gaussian (σ·√d) for the per-coordinate standard deviation σ in the Gaussian RDP
  formula. They also clipped each token independently while accounting as if the
  complete released embedding had radius R. Both choices understated ε. This module
  now clips each complete released embedding once and accounts with the actual
  per-coordinate standard deviation. The historical ε=0.1994 claim is invalid.

  This module provides:
    1. L2 clipping (makes Δ₂f bounded = 2R)
    2. RDP accountant (computes formal (ε, δ) from σ, R, α)
    3. Optional 8-bit quantization (extra defense + 4× bandwidth reduction)

No retraining is required: this is a client-side governed release mechanism;
the encoder checkpoint remains unchanged.

References:
  - Dwork & Roth 2014, "The Algorithmic Foundations of Differential Privacy"
  - Mironov 2017, "Rényi Differential Privacy" (CCS)
  - Abadi et al. 2016, "Deep Learning with Differential Privacy" (DP-SGD)
  - Balle & Wang 2018, "Improving the Gaussian Mechanism for Differential
    Privacy" (Analytic Gaussian, ICML)
  - Xia et al. CVPR 2025, "CEM" — H(X|T) lower bound on any MIA reconstruction MSE
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

# Default RDP alpha order list (Mironov 2017 + Abadi et al. 2016 recommended).
# Covers the typical optimal alpha range for Gaussian mechanism on embedding release.
DEFAULT_ALPHAS: List[float] = (
    [1 + x / 10.0 for x in range(1, 100)]  # 1.1 to 10.9 (fine-grained)
    + list(range(12, 64))                   # 12 to 63 (coarse)
)


def _log_addexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def rdp_gaussian(alpha: float, sigma_per_coord: float, delta_2f: float = 2.0) -> float:
    """RDP of the Gaussian mechanism at order alpha.

    For release f(D) + N(0, σ²I_d) with L2 sensitivity Δ₂f:
        RDP(α) = α · (Δ₂f)² / (2 σ²)

    Args:
        alpha: RDP order (> 1).
        sigma_per_coord: Standard deviation of every coordinate of the isotropic
            Gaussian mechanism. This is the σ in N(0, σ²I).
        delta_2f: L2 sensitivity (default 2.0 for ||f|| <= R=1 clipping).

    Returns:
        RDP value at order alpha.
    """
    if alpha <= 1.0:
        raise ValueError(f"alpha must be > 1, got {alpha}")
    if sigma_per_coord <= 0:
        return float("inf")
    return alpha * (delta_2f ** 2) / (2.0 * sigma_per_coord ** 2)


def rdp_to_dp(
    rdp_values: List[float],
    alphas: List[float],
    delta: float,
) -> Tuple[float, float]:
    """Convert RDP values to (ε, δ)-DP.

    ε = min_α [ RDP(α) + ln(1/δ) / (α - 1) ]

    Args:
        rdp_values: RDP at each alpha order.
        alphas: The alpha orders (same length as rdp_values).
        delta: Target δ (e.g., 1e-5).

    Returns:
        (epsilon, optimal_alpha) — the tightest ε and the alpha that achieved it.
    """
    if len(rdp_values) != len(alphas):
        raise ValueError("rdp_values and alphas must have same length")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    eps_opt = float("inf")
    alpha_opt = alphas[0]
    log_inv_delta = math.log(1.0 / delta)
    for rdp, alpha in zip(rdp_values, alphas):
        if alpha <= 1.0:
            continue
        eps = rdp + log_inv_delta / (alpha - 1.0)
        if eps < eps_opt:
            eps_opt = eps
            alpha_opt = alpha
    return eps_opt, alpha_opt


def compute_dp_guarantee(
    sigma_per_coord: float,
    dim: int,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
    alphas: Optional[List[float]] = None,
) -> Tuple[float, float, float]:
    """Compute formal (ε, δ)-DP guarantee for Gaussian noise on embeddings.

    Args:
        sigma_per_coord: Per-coordinate Gaussian std (the σ you actually add).
        dim: Embedding dimensionality (e.g., 16 * 2560 = 40960).
        delta: Target δ (default 1e-5, standard for medical data).
        clip_norm: L2 norm the embedding is clipped to before noise (R).
            Sensitivity Δ₂f = 2 * clip_norm. Default 1.0 (standard).
        alphas: RDP orders to try (default DEFAULT_ALPHAS).

    Returns:
        (epsilon, optimal_alpha, sigma_l2):
            epsilon — the tightest ε satisfying (ε, δ)-DP
            optimal_alpha — the α achieving it
            sigma_l2 — the L2 norm of the noise (= sigma_per_coord * sqrt(dim))
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    # L2 norm of isotropic Gaussian noise N(0, σ²I_d) concentrates around σ·√d
    sigma_l2 = sigma_per_coord * math.sqrt(dim)
    delta_2f = 2.0 * clip_norm
    rdp_values = [rdp_gaussian(a, sigma_per_coord, delta_2f) for a in alphas]
    eps, alpha_opt = rdp_to_dp(rdp_values, alphas, delta)
    return eps, alpha_opt, sigma_l2


def sigma_for_target_epsilon(
    target_epsilon: float,
    dim: int,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
    alphas: Optional[List[float]] = None,
) -> Tuple[float, float, float]:
    """Find the minimum per-coordinate σ achieving (target_ε, δ)-DP.

    Binary search over the per-coordinate σ to find the smallest σ satisfying
    the target ε.
    Useful for picking CFP noise level from a desired formal DP guarantee.

    Args:
        target_epsilon: Desired ε (e.g., 1.0).
        dim: Embedding dimensionality.
        delta: Target δ.
        clip_norm: L2 clipping norm R.
        alphas: RDP orders.

    Returns:
        (sigma_per_coord, sigma_l2, optimal_alpha):
            The minimum per-coordinate σ (and L2 norm) achieving the target.
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    delta_2f = 2.0 * clip_norm
    # Binary search on the per-coordinate Gaussian standard deviation.
    lo, hi = 1e-3, 1e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        rdp_values = [rdp_gaussian(a, mid, delta_2f) for a in alphas]
        eps, _ = rdp_to_dp(rdp_values, alphas, delta)
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
    sigma_per_coord = 0.5 * (lo + hi)
    sigma_l2 = sigma_per_coord * math.sqrt(dim)
    # Recompute optimal alpha at the final sigma
    rdp_values = [rdp_gaussian(a, sigma_per_coord, delta_2f) for a in alphas]
    _, alpha_opt = rdp_to_dp(rdp_values, alphas, delta)
    return sigma_per_coord, sigma_l2, alpha_opt


def l2_clip(features: torch.Tensor, clip_norm: float = 1.0) -> torch.Tensor:
    """L2 clip features to ||f||₂ ≤ clip_norm (in-place safe, returns new tensor).

    This is the SENSITIVITY-BOUNDING step that makes DP analysis possible.
    Without clipping, Δ₂f is unbounded and no formal DP guarantee holds.

    Args:
        features: [..., D] tensor.
        clip_norm: Target L2 norm R. Default 1.0.

    Returns:
        Clipped features (same shape).
    """
    if clip_norm <= 0:
        return features
    # A 3-D tensor is a batch of complete [tokens, channels] releases. A 2-D
    # tensor is one complete release, not a batch of independently clipped tokens.
    if features.dim() >= 3:
        flat = features.reshape(features.shape[0], -1)
    else:
        flat = features.reshape(1, -1)
    norms = flat.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (clip_norm / norms).clamp(max=1.0)  # only scale down
    flat_clipped = flat * scale
    return flat_clipped.reshape_as(features)


def quantize_int8(
    features: torch.Tensor, *, value_range: Optional[float] = None
) -> torch.Tensor:
    """8-bit symmetric quantization (extra privacy defense + 4× bandwidth reduction).

    Quantization adds ~Uniform(-2^-9, 2^-9) noise per coordinate, which:
      - Provides ~20-30% additional inversion-attack resistance (Vec2Text experiments)
      - Reduces client→server bandwidth from 160 KB to 40 KB (16 × 2560 × 1 byte)
      - Is fully compatible with downstream Perceiver/LLM (they dequantize transparently)

    Symmetric int8: maps [-abs_max, abs_max] → [-127, 127].
    Dequantization recovers float32 with ~1/256 relative error.

    Args:
        features: [..., D] float tensor.

    Returns:
        Dequantized float tensor (same shape, dtype float32) — the simulated
        effect of int8 quantize→dequantize round-trip.
    """
    # The historical default derives the scale from the current record.  That
    # mode is useful for ablations but must not be used for an accounted release.
    if value_range is None:
        abs_max = features.abs().max().clamp(min=1e-8)
    else:
        if not math.isfinite(float(value_range)) or value_range <= 0:
            raise ValueError("value_range must be finite and positive")
        abs_max = features.new_tensor(float(value_range))
    scale = 127.0 / abs_max
    q = torch.round(features.clamp(-abs_max, abs_max) * scale).clamp(-127, 127)
    return (q / scale).to(features.dtype)


class DPCFPAccountant:
    """Formal (ε, δ)-DP accountant for governed embedding release.

    The mechanism combines:
      1. L2 clipping (bounds sensitivity to 2R)
      2. RDP accounting (computes formal ε from per-coordinate σ and R)
      3. Optional int8 quantization (extra defense + bandwidth reduction)
      4. v170 FIX: Anisotropic noise ACTIVELY APPLIED when diagnostic_sensitivity
         is provided (was previously claimed but only isotropic was applied).

    Usage (client side, replaces direct CFP.apply()):
        from fpavlm.cipher_native.dp_accountant import DPCFPAccountant

        accountant = DPCFPAccountant(
            noise_fraction=24.0,    # v165 canonical anisotropic σ
            dim=16 * 2560,          # embedding dimensionality
            clip_norm=1.0,          # L2 clip to ||f|| <= 1 (sensitivity = 2)
            delta=1e-5,             # standard for medical data
            quantize=True,          # extra int8 defense
            anisotropy_lambda=1.0,  # v164 Innovation B
            diagnostic_sensitivity=sensitivity_tensor,  # [2560] per-dim
        )

        clean_features = client_engine.decrypt(ciphertexts)  # [16, 2560]
        noised_features, dp_report = accountant.apply_and_account(clean_features)

        # dp_report contains:
        #   epsilon: float  — formal (ε, δ)-DP guarantee (conservative)
        #   delta: float
        #   alpha_opt: float  — RDP order achieving the tightest ε
        #   sigma_l2: float  — L2 norm of the noise (minimum per-dim, conservative)
        #   sigma_per_coord: float  — per-coordinate σ (minimum, conservative)
        #   clip_norm: float
        #   quantized: bool
        #   anisotropy_active: bool  — v170: TRUE when anisotropic noise applied
        print(f"Privacy: ({dp_report.epsilon:.3f}, {dp_report.delta:.1e})-DP")

        client.send_to_server(noised_features)

    ANISOTROPIC NOISE (v164 Innovation B, v170 FIX — ACTIVELY APPLIED):
      When diagnostic_sensitivity is provided (per-dim [llm_dim] tensor) and
      anisotropy_lambda > 0, noise becomes ANISOTROPIC per-dimension:
          σ_d = sigma_per_coord * (1 + λ * s_d)
      where s_d is the normalized sensitivity in [0, 1]. Diagnostic-sensitive
      dimensions (high s_d, lesion-bearing) get MORE noise (stronger privacy);
      background dimensions (low s_d) get less (preserved utility). This is a
      Pareto improvement over isotropic: same average noise budget → higher
      utility + equal/better privacy.

      DP ANALYSIS (conservative bound): for anisotropic Gaussian noise, the
      anisotropic noise, we use the MINIMUM per-dim σ (= sigma_per_coord * 1.0
      when s_d=0, i.e., the background dim with least noise) as the conservative
      σ_per_coord in the RDP formula. The reported ε is thus a VALID UPPER BOUND
      for ALL dimensions — actual privacy on diagnostic dims (with higher σ) is
      STRICTLY STRONGER. The historical ε≈0.1994 result was caused by using σ√d
      in place of σ and is not a valid bound; the default corrected value is much
      larger and must not be described as strong privacy.
    """

    def __init__(
        self,
        noise_fraction: float,
        dim: int,
        feature_std: Optional[float] = None,
        target_epsilon: Optional[float] = None,
        clip_norm: float = 1.0,
        delta: float = 1e-5,
        alphas: Optional[List[float]] = None,
        quantize: bool = False,
        anisotropy_lambda: float = 0.0,  # 0 = isotropic
        diagnostic_sensitivity: Optional[torch.Tensor] = None,  # v170 FIX
        renormalize_sensitivity: bool = True,
    ):
        self.noise_fraction = noise_fraction
        self.dim = dim
        self.feature_std = feature_std
        if dim < 1:
            raise ValueError("dim must be positive")
        if noise_fraction < 0:
            raise ValueError("noise_fraction must be non-negative")
        if feature_std is not None and feature_std <= 0:
            raise ValueError("feature_std must be positive when provided")
        if target_epsilon is not None and target_epsilon <= 0:
            raise ValueError("target_epsilon must be positive")
        self.target_epsilon = target_epsilon
        self.clip_norm = clip_norm
        self.delta = delta
        if clip_norm <= 0:
            raise ValueError("clip_norm must be positive")
        if not 0 < delta < 1:
            raise ValueError("delta must lie in (0, 1)")
        if anisotropy_lambda < 0:
            raise ValueError("anisotropy_lambda must be non-negative")
        self.alphas = alphas if alphas is not None else DEFAULT_ALPHAS
        self.quantize = quantize
        self.anisotropy_lambda = anisotropy_lambda

        # v170 FIX: Actually implement anisotropic CFP (was claimed but never
        # applied by historical baselines). The deployed rule is:
        # per-dim σ scaled by diagnostic sensitivity. Dimensions carrying
        # diagnostic signal (lesion features) get MORE noise (stronger privacy);
        # background dims get less (preserved utility).
        if diagnostic_sensitivity is not None and anisotropy_lambda > 0:
            s = diagnostic_sensitivity.float()
            if s.ndim != 1:
                raise ValueError("diagnostic_sensitivity must be one-dimensional")
            if dim % s.numel() != 0:
                raise ValueError(
                    "diagnostic_sensitivity length must divide the complete release dimension"
                )
            if not torch.isfinite(s).all():
                raise ValueError("diagnostic_sensitivity must contain only finite values")
            if s.min() < 0:
                raise ValueError("diagnostic_sensitivity must be non-negative")
            if renormalize_sensitivity:
                s_min, s_max = s.min(), s.max()
                if s_max > s_min:
                    s = (s - s_min) / (s_max - s_min)  # normalize to [0, 1]
                else:
                    s = torch.zeros_like(s)
            else:
                s = s.clamp(0.0, 1.0)
            self.diagnostic_sensitivity = s
            self._per_dim_scale = 1.0 + anisotropy_lambda * s  # [llm_dim]
        else:
            self.diagnostic_sensitivity = None
            self._per_dim_scale = None

    def apply_and_account(
        self,
        features: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, "DPReport"]:
        """Apply L2 clip → Gaussian noise → (optional) int8 quantize, and compute DP.

        Args:
            features: [B, 16, 2560] or [16, 2560] clean decrypted features.
            generator: Optional torch.Generator for reproducibility.

        Returns:
            (noised_features, dp_report):
                noised_features — same shape as input, ready to send to server.
                dp_report — DPReport with formal (ε, δ) guarantee.
        """
        if features.dim() not in (2, 3):
            raise ValueError("features must have shape [tokens, channels] or [batch, tokens, channels]")
        # Step 1: L2 clip to bound sensitivity (REQUIRED for DP analysis)
        if features.shape[-2] * features.shape[-1] != self.dim:
            raise ValueError(
                f"features contain {features.numel()} values, incompatible with release dim {self.dim}"
            )
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite before privacy accounting")
        if (
            self._per_dim_scale is not None
            and features.shape[-1] != self._per_dim_scale.numel()
        ):
            raise ValueError(
                "diagnostic_sensitivity length must match the feature channel dimension"
            )
        clipped = l2_clip(features, self.clip_norm)

        # Step 2: Determine sigma from public configuration only. A noise scale
        # estimated from the current private release is data-dependent and does
        # not define the fixed Gaussian mechanism accounted for below.
        std = self.feature_std
        if std is None:
            std = self.clip_norm / math.sqrt(self.dim)
        if self.target_epsilon is not None:
            sigma_per_coord, _, _ = sigma_for_target_epsilon(
                target_epsilon=self.target_epsilon,
                dim=self.dim,
                delta=self.delta,
                clip_norm=self.clip_norm,
                alphas=self.alphas,
            )
        else:
            sigma_per_coord = self.noise_fraction * std

        # Step 3: Add Gaussian noise. v170 FIX: ANISOTROPIC when
        # diagnostic_sensitivity is provided. Per-dimension:
        # sigma_d = sigma_per_coord * (1 + lambda * s_d).
        # Diagnostic-sensitive dims (high s_d) get MORE noise (stronger privacy
        # on lesion features); background dims get less (preserved utility).
        # DP analysis uses MINIMUM per-dim σ (= sigma_per_coord when s_d=0) as
        # conservative bound (worst-case over dimensions = the dim with least
        # noise). The reported ε is thus a VALID UPPER BOUND for ALL dimensions;
        # actual privacy on diagnostic dims (higher σ) is STRICTLY STRONGER.
        if self._per_dim_scale is not None:
            scale = self._per_dim_scale.to(features.device)
            sigma_per_dim = sigma_per_coord * scale  # [llm_dim]
            if features.dim() == 3:
                # [B, 16, 2560] → sigma [1, 1, 2560]
                sigma_broadcast = sigma_per_dim.view(1, 1, -1)
            else:
                # [16, 2560] → sigma [1, 2560]
                sigma_broadcast = sigma_per_dim.view(1, -1)
            noise = torch.randn(
                features.shape, generator=generator,
                dtype=features.dtype, device=features.device,
            ) * sigma_broadcast
            anisotropy_active = True
            # Conservative DP bound: MIN per-dim σ (= base σ at s_d=0)
            sigma_per_coord_for_dp = sigma_per_coord
        else:
            # Isotropic noise (backward compatible; no sensitivity provided)
            noise = torch.randn(
                features.shape, generator=generator,
                dtype=features.dtype, device=features.device,
            ) * sigma_per_coord
            anisotropy_active = False
            sigma_per_coord_for_dp = sigma_per_coord
        noised = clipped + noise

        # Step 4: Optional int8 quantization (extra defense + bandwidth)
        if self.quantize:
            # Fixed public range is post-processing of the Gaussian release;
            # a per-record max would publish an additional private statistic.
            noised = quantize_int8(noised, value_range=self.clip_norm)

        # Step 5: Compute formal (ε, δ)-DP via RDP accounting.
        # Uses the conservative (minimum) per-dim σ — valid upper bound for
        # all dimensions when anisotropy is active.
        epsilon, alpha_opt, sigma_l2 = compute_dp_guarantee(
            sigma_per_coord=sigma_per_coord_for_dp,
            dim=self.dim,
            delta=self.delta,
            clip_norm=self.clip_norm,
            alphas=self.alphas,
        )

        report = DPReport(
            epsilon=epsilon,
            delta=self.delta,
            alpha_opt=alpha_opt,
            sigma_per_coord=sigma_per_coord_for_dp,
            sigma_l2=sigma_l2,
            clip_norm=self.clip_norm,
            quantized=self.quantize,
            noise_fraction=self.noise_fraction,
            feature_std=std,
            dim=self.dim,
            anisotropy_active=anisotropy_active,
            anisotropy_lambda=self.anisotropy_lambda if anisotropy_active else 0.0,
            quantization_range=self.clip_norm if self.quantize else None,
        )
        return noised, report

    def __repr__(self) -> str:
        return (
            f"DPCFPAccountant(noise_fraction={self.noise_fraction}, "
            f"dim={self.dim}, clip_norm={self.clip_norm}, "
            f"delta={self.delta}, quantize={self.quantize})"
        )


class DPReport:
    """Formal DP guarantee report (serializable)."""

    def __init__(
        self,
        epsilon: float,
        delta: float,
        alpha_opt: float,
        sigma_per_coord: float,
        sigma_l2: float,
        clip_norm: float,
        quantized: bool,
        noise_fraction: float,
        feature_std: float,
        dim: int,
        anisotropy_active: bool = False,  # v170 FIX: track anisotropy status
        anisotropy_lambda: float = 0.0,
        quantization_range: Optional[float] = None,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.alpha_opt = alpha_opt
        self.sigma_per_coord = sigma_per_coord
        self.sigma_l2 = sigma_l2
        self.clip_norm = clip_norm
        self.quantized = quantized
        self.noise_fraction = noise_fraction
        self.feature_std = feature_std
        self.dim = dim
        self.anisotropy_active = anisotropy_active
        self.anisotropy_lambda = anisotropy_lambda
        self.quantization_range = quantization_range

    def __repr__(self) -> str:
        aniso_str = f", aniso=ON(lambda={self.anisotropy_lambda:.1f})" if self.anisotropy_active else ""
        return (
            f"DPReport(epsilon={self.epsilon:.4f}, delta={self.delta:.1e}, "
            f"alpha*={self.alpha_opt:.1f}, sigma_L2={self.sigma_l2:.3f}, "
            f"clip_R={self.clip_norm}, int8={self.quantized}{aniso_str})"
        )

    def composed_epsilon(self, releases: int) -> float:
        """Return a conservative basic-composition epsilon for repeated releases.

        The accountant models one complete image release.  Unless a tighter
        moments accountant is supplied for the query policy, ``m`` releases
        must be reported with at least ``m * epsilon``.
        """
        if isinstance(releases, bool) or not isinstance(releases, int) or releases < 1:
            raise ValueError("releases must be a positive integer")
        return float(releases) * float(self.epsilon)

    @property
    def contains_image_information(self) -> bool:
        """Plaintext releases remain utility-bearing image information channels."""
        return True

    def to_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "alpha_optimal": self.alpha_opt,
            "sigma_per_coord": self.sigma_per_coord,
            "sigma_l2": self.sigma_l2,
            "clip_norm": self.clip_norm,
            "quantized": self.quantized,
            "noise_fraction": self.noise_fraction,
            "feature_std": self.feature_std,
            "dim": self.dim,
            "anisotropy_active": self.anisotropy_active,
            "anisotropy_lambda": self.anisotropy_lambda,
            "quantization_range": self.quantization_range,
            "guarantee_str": f"({self.epsilon:.4f}, {self.delta:.1e})-DP",
        }


__all__ = [
    "DEFAULT_ALPHAS",
    "rdp_gaussian",
    "rdp_to_dp",
    "compute_dp_guarantee",
    "sigma_for_target_epsilon",
    "l2_clip",
    "quantize_int8",
    "DPCFPAccountant",
    "DPReport",
    "cli_privacy_check",
]


# =============================================================================
# CLI Entry Point (installed as `fpavlm-privacy-check` via [project.scripts])
# =============================================================================

def cli_privacy_check():
    """CLI entry point for `fpavlm-privacy-check` command.

    Computes the formal (ε, δ)-DP guarantee for a given CFP configuration.

    Usage:
        fpavlm-privacy-check --sigma-frac 24.0 --dim 40960 --delta 1e-5
        fpavlm-privacy-check --target-eps 1.0 --dim 40960  # find required σ
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog="fpavlm-privacy-check",
        description="Compute formal (epsilon, delta)-DP guarantee for FHE-VLM CFP",
    )
    parser.add_argument("--sigma-frac", type=float, default=None,
                        help="CFP noise fraction (sigma * feature_std). If set, compute epsilon.")
    parser.add_argument("--target-eps", type=float, default=None,
                        help="Target epsilon. If set, compute required sigma.")
    parser.add_argument("--dim", type=int, default=16 * 2560,
                        help="Embedding dimensionality (default 16*2560=40960)")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="DP delta (default 1e-5)")
    parser.add_argument("--clip-norm", type=float, default=1.0,
                        help="L2 clip norm R (default 1.0, sensitivity = 2R)")
    parser.add_argument("--feature-std", type=float, default=0.005,
                        help="Per-coordinate feature std (default 0.005, ~1/sqrt(dim))")
    args = parser.parse_args()

    print("=" * 70)
    print("FHE-VLM CFP Formal DP Guarantee")
    print("=" * 70)
    print(f"Embedding dim:       {args.dim}")
    print(f"Delta:               {args.delta:.1e}")
    print(f"L2 clip norm (R):    {args.clip_norm}  (sensitivity Delta_2 f = {2*args.clip_norm})")
    print(f"Feature std:         {args.feature_std:.6f}")
    print("-" * 70)

    if args.sigma_frac is not None:
        sigma_per_coord = args.sigma_frac * args.feature_std
        eps, alpha_opt, sigma_l2 = compute_dp_guarantee(
            sigma_per_coord=sigma_per_coord,
            dim=args.dim,
            delta=args.delta,
            clip_norm=args.clip_norm,
        )
        print(f"Sigma (per-coord):   {sigma_per_coord:.6f}")
        print(f"Sigma (L2 norm):     {sigma_l2:.4f}")
        print(f"Optimal alpha:       {alpha_opt:.2f}")
        print(f"Formal epsilon:      {eps:.4f}")
        print("-" * 70)
        if eps < 1.0:
            print("Numerically strong DP bound: epsilon < 1; validate the threat model before deployment")
        elif eps < 3.0:
            print("Moderate numerical DP bound: 1 <= epsilon < 3; validate deployment suitability")
        else:
            print("Weak DP bound: epsilon >= 3; do not claim strong embedding privacy")

    elif args.target_eps is not None:
        sigma_per_coord, sigma_l2, alpha_opt = sigma_for_target_epsilon(
            target_epsilon=args.target_eps,
            dim=args.dim,
            delta=args.delta,
            clip_norm=args.clip_norm,
        )
        sigma_frac = sigma_per_coord / args.feature_std
        print(f"Target epsilon:      {args.target_eps}")
        print(f"Required sigma (L2): {sigma_l2:.4f}")
        print(f"Required sigma (per-coord): {sigma_per_coord:.6f}")
        print(f"Required sigma_frac: {sigma_frac:.2f} * feature_std")
        print(f"Optimal alpha:       {alpha_opt:.2f}")

    else:
        # Default: show table of sigma_frac -> epsilon mappings.
        print(f"\n{'sigma_frac':>10} {'sigma_coord':>15} {'sigma_L2':>10} "
              f"{'alpha*':>8} {'epsilon':>10}  Verdict")
        print("-" * 70)
        for sigma_frac in [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0]:
            sigma_per_coord = sigma_frac * args.feature_std
            eps, alpha_opt, sigma_l2 = compute_dp_guarantee(
                sigma_per_coord=sigma_per_coord,
                dim=args.dim,
                delta=args.delta,
                clip_norm=args.clip_norm,
            )
            if eps < 1.0:
                verdict = "STRONG"
            elif eps < 3.0:
                verdict = "moderate"
            else:
                verdict = "weak"
            print(f"{sigma_frac:>10.1f} {sigma_per_coord:>15.6f} {sigma_l2:>10.4f} "
                  f"{alpha_opt:>8.2f} {eps:>10.4f}  {verdict}")

    print("=" * 70)
