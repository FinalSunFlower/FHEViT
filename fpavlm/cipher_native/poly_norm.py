"""PolyNorm: Polynomial LayerNorm Replacement for FHE-Native Medical ViT.

v117 INNOVATION: Anatomy-Aware Polynomial Normalization (AAPN)

  v116 limitation: All tokens use SHARED polynomial coefficients (mean across D).
    This is medically suboptimal because different anatomical regions have
    different feature distributions (lung fields are radiolucent/dark, cardiac
    silhouette is radiopaque/bright). A single normalization polynomial cannot
    optimally normalize all regions.

  v117 solution: Per-REGION polynomial coefficients (c0_k, c1_k, c2_k for
    region k). Each token uses its region's coefficients based on FIXED
    anatomical position (not data-dependent routing).

  Why this is FHE-native:
    - Each token is a SEPARATE ciphertext in v116/v117
    - polyval can apply DIFFERENT coefficients to each ciphertext
    - ZERO extra FHE depth (same polyval operation, just different coeffs)
    - NO rotate, NO conditional branch (FHE-safe)

  Why this is medical-specific:
    - 5 anatomical regions for chest X-ray (heart, L lung, R lung, etc.)
    - Region assignment based on 2D position (fixed anatomical prior)
    - Each region learns its own normalization polynomial

FIRST-PRINCIPLES ANALYSIS (why v116 needs PolyNorm):

  Standard LayerNorm: y = γ · (x - μ) / sqrt(var + ε) + β
    - Requires 1/sqrt(var), which has SINGULARITY at var=0
    - v111 failed (cos=0.862) because polynomial approximation of 1/sqrt diverged
      when var was small (coefficient explosion)

  v116/v117 PolyNorm solution (adapted from ULD-Net, ICLR 2026):
    Instead of approximating 1/sqrt(var) separately, approximate the WHOLE
    normalization function as a polynomial in the input distribution.

    z_norm = c0 + c1·x + c2·x²        (degree-2 polynomial normalization)
    y = γ · z_norm + β                 (affine, fused into next matmul)

  FHE implementation (uses TenSEAL's polyval primitive):
    - v116: z_norm = ct.polyval([c0_shared, c1_shared, c2_shared])  (shared coeffs)
    - v117: z_norm = ct[i].polyval([c0_regions[r(i)], c1_regions[r(i)], c2_regions[r(i)]])
            where r(i) is the region of token i (per-token coeffs)

  Total depth: 2 (same as v116 — per-region coeffs are FREE)

  Privacy:
    - All computation on SERVER side (ciphertexts)
    - Coefficients are model parameters (server-side)
    - Client never sees coefficients
    - Server never sees plaintext features

  Literature:
    - ULD-Net (ICLR 2026): Fully polynomial ViT, depth ≤3 per operator
    - PolyNeXt (2026): Polynomial MetaFormer backbones
    - v111 failure (this codebase): LayerNorm polynomial approximation diverged
    - v117 AAPN: Per-region polynomial normalization (NEW, medical-specific)
"""
from __future__ import annotations

import math
from typing import List, Optional, Union

import torch
import torch.nn as nn

from .ciphertext_tensor import CiphertextTensor


class PolyNorm(nn.Module):
    """Polynomial LayerNorm replacement for FHE-native medical ViT.

    v116: SHARED polynomial coefficients (mean across D) for all tokens.
    v117: PER-REGION polynomial coefficients (c0_k, c1_k, c2_k for region k).

    Replaces standard LayerNorm with a learned degree-2 polynomial:
        z_norm = c0 + c1*x + c2*x^2          (per-region in v117)
        y = gamma * z_norm + beta             (fused into next matmul)

    FHE depth: 2 (polyval degree-2) — SAME for v116 and v117 (per-region is FREE)

    Args:
        dim: Feature dimension D
        eps: Small constant for numerical stability (not used in polynomial form)
        init_scale: Initial scale for c1 (default: 1/sqrt(D), matching LayerNorm)
        poly_degree: Polynomial degree (always 2 — v124 LPPN uses degree-2 PolyNorm
            with degree-1 PolyAct). Kept as parameter for API stability.
        num_regions: Number of anatomical regions (1=v116 shared, >1=v117 per-region)
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        init_scale: Optional[float] = None,
        poly_degree: int = 2,
        num_regions: int = 1,
        channelwise_coeffs: bool = False,
        medical_prior: Optional[dict] = None,
        symmetry_pairs: Optional[List[tuple]] = None,
    ):
        """Initialize PolyNorm.

        Args:
            dim: Feature dimension D.
            eps: Numerical stability (unused in polynomial form).
            init_scale: Initial scale for c1 (default 1/sqrt(D)).
            poly_degree: Always 2 (v124 LPPN).
            num_regions: 1=v116 shared, >1=v117 per-region.
            medical_prior: v155 MedPolyNorm — dict with key "region_stats":
                {"region_stats": [{"mean": μ_r, "std": σ_r}, ...], }.
                When provided, per-region coefficients are initialized from
                medical statistics: c0_r=-μ_r/σ_r, c1_r=1/σ_r, c2_r=α (lesion
                sensitivity). This replaces the generic zero-init with a
                medical-meaningful starting point. If None, falls back to
                generic init (backward compatible).
            symmetry_pairs: v155 — list of (r_a, r_b) contralateral region
                pairs. When provided, paired regions share coefficients (r_b
                copies r_a's init). Encodes the chest X-ray left-right symmetry
                prior. If None, no symmetry constraint (backward compatible).
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.poly_degree = poly_degree
        self.num_regions = num_regions
        self.channelwise_coeffs = bool(channelwise_coeffs and num_regions > 1)

        # Learnable affine parameters (like LayerNorm's gamma, beta)
        # These are FUSED into the next matmul in cipher path (0 FHE depth)
        self.weight = nn.Parameter(torch.ones(dim))  # gamma
        self.bias = nn.Parameter(torch.zeros(dim))   # beta

        if init_scale is None:
            init_scale = 1.0 / math.sqrt(dim)

        if num_regions == 1:
            # v116 compat: per-channel coeffs (cipher uses mean across D)
            self.c0 = nn.Parameter(torch.zeros(dim))
            self.c1 = nn.Parameter(torch.full((dim,), init_scale))
            self.c2 = nn.Parameter(torch.zeros(dim))
            # v117 per-region params (None for v116 compat)
            self.c0_regions = None
            self.c1_regions = None
            self.c2_regions = None
        else:
            # v117: per-region SCALAR coeffs (matches cipher path exactly)
            # v155 MedPolyNorm: initialize from medical statistics if provided
            coeff_shape = (num_regions, dim) if self.channelwise_coeffs else (num_regions,)
            if medical_prior is not None and "region_stats" in medical_prior:
                stats = medical_prior["region_stats"]
                assert len(stats) == num_regions, (
                    f"medical_prior has {len(stats)} regions, expected {num_regions}"
                )
                # MedPolyNorm init: c0=-μ/σ (center), c1=1/σ (scale),
                # c2=α (lesion sensitivity). α is a small positive value that
                # amplifies |x-μ| (abnormal/lesion) via the x² term while
                # suppressing |x-μ|≈0 (normal). This is the lesion-sensitive
                # quadratic — the medical-specific contribution.
                c0_init = torch.zeros(coeff_shape)
                c1_init = torch.zeros(coeff_shape)
                c2_init = torch.zeros(coeff_shape)
                for r in range(num_regions):
                    mu_r = stats[r].get("mean", 0.0)
                    sigma_r = stats[r].get("std", 1.0)
                    sigma_r = max(sigma_r, 1e-6)  # avoid div-by-zero
                    c0_init[r] = -mu_r / sigma_r
                    c1_init[r] = 1.0 / sigma_r
                    # Lesion sensitivity: α = 0.1 / σ_r. Small positive so the
                    # x² term amplifies outliers (lesions) but doesn't dominate
                    # normal features. Tunable via fine-tuning.
                    c2_init[r] = 0.1 / sigma_r

                # v155 symmetry constraint: paired regions share coefficients.
                # This encodes the chest X-ray left-right symmetry prior —
                # contralateral lung regions should normalize identically.
                if symmetry_pairs is not None:
                    for r_a, r_b in symmetry_pairs:
                        c0_init[r_b] = c0_init[r_a]
                        c1_init[r_b] = c1_init[r_a]
                        c2_init[r_b] = c2_init[r_a]

                self.c0_regions = nn.Parameter(c0_init)
                self.c1_regions = nn.Parameter(c1_init)
                self.c2_regions = nn.Parameter(c2_init)
            else:
                # Generic init (backward compatible): c0=0, c1=1/sqrt(D), c2=0
                self.c0_regions = nn.Parameter(torch.zeros(coeff_shape))
                self.c1_regions = nn.Parameter(torch.full(coeff_shape, init_scale))
                self.c2_regions = nn.Parameter(torch.zeros(coeff_shape))
            # v116 per-channel params (None for v117)
            self.c0 = None
            self.c1 = None
            self.c2 = None

        # Cached cipher-path coefficients (set by prepare_cipher_weights)
        # v116: scalars (mean across D)
        # v117: lists of R scalars
        self._cipher_c0: Optional[Union[float, List[float]]] = None
        self._cipher_c1: Optional[Union[float, List[float]]] = None
        self._cipher_c2: Optional[Union[float, List[float]]] = None

    def forward_plain(
        self,
        x: torch.Tensor,
        token_regions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Plain forward: x is [B, N, D] or [N, D] tensor.

        v116 (num_regions==1): Uses per-channel coeffs (c0, c1, c2 are [D]).
        v117 (num_regions>1): Uses per-region SCALAR coeffs (matches cipher path).
            token_regions: [N] LongTensor of region indices (0..R-1).

        Computes: y = gamma * z_norm + beta
        """
        if self.num_regions == 1 or token_regions is None:
            # v116 path: per-channel coeffs (cipher uses mean)
            # NOTE: This creates a train-inference gap (plain uses per-channel,
            # cipher uses mean). v116 smoke test tolerated this.
            z_norm = self.c0 + self.c1 * x + self.c2 * x * x
        else:
            # v117 path: per-region SCALAR coeffs (NO train-inference gap)
            # token_regions: [N] → gather per-token coeffs
            c0_tok = self.c0_regions[token_regions]
            c1_tok = self.c1_regions[token_regions]  # [N]
            c2_tok = self.c2_regions[token_regions]  # [N]
            # Reshape for broadcasting over [B, N, D]
            # x is [B, N, D], coeffs need to be [1, N, 1] or [N, 1]
            if self.channelwise_coeffs:
                if x.dim() == 3:
                    c0_tok = c0_tok.unsqueeze(0)
                    c1_tok = c1_tok.unsqueeze(0)
                    c2_tok = c2_tok.unsqueeze(0)
            elif x.dim() == 3:
                c0_tok = c0_tok.view(1, -1, 1)
                c1_tok = c1_tok.view(1, -1, 1)
                c2_tok = c2_tok.view(1, -1, 1)
            else:
                c0_tok = c0_tok.view(-1, 1)
                c1_tok = c1_tok.view(-1, 1)
                c2_tok = c2_tok.view(-1, 1)
            z_norm = c0_tok + c1_tok * x + c2_tok * x * x

        # Apply affine (gamma, beta) — in cipher path these are fused into next matmul
        y = self.weight * z_norm + self.bias
        return y

    def forward_cipher(
        self,
        x: CiphertextTensor,
        token_regions: Optional[List[int]] = None,
        x_sq: Optional[CiphertextTensor] = None,
        z_norm: Optional[CiphertextTensor] = None,
    ) -> CiphertextTensor:
        """Cipher forward: x is CiphertextTensor with N ciphers, each D-slot.

        v116 (num_regions==1): SHARED coeffs (mean across D) for all tokens.
        v117 (num_regions>1): PER-REGION coeffs — each token uses its region's coeffs.

        v152 CAS (Client-Assisted Squaring): If x_sq is provided (fresh encryption
        of x² from client), compute z = c0 + c1·x + c2·x_sq using ONLY cipher×
        plain ops (depth +1). This ELIMINATES the cipher×cipher in c2·x² (the
        dominant CKKS noise source). Without x_sq, falls back to polyval (depth+2).

        v165 CADN compat: z_norm is accepted but IGNORED by PolyNorm (PolyNorm does
        not use CADN). This allows the same forward_cipher call site to work for
        both PolyNorm and CADNLayer (CADNLayer uses z_norm, PolyNorm uses x_sq).
        When use_cadn=True at the block level, CADNLayer is used instead of PolyNorm,
        so this code path is never reached in v165 deployment.

        FHE implementation:
          v116/v117 (no CAS): z_norm = ct.polyval([c0, c1, c2])  (depth +2)
          v152 CAS:           z_norm = c0 + c1·ct + c2·ct_sq     (depth +1)
            where ct_sq is a FRESH ciphertext of x² from the client

        Returns: CiphertextTensor at depth (x.depth + 2) without CAS, or
                 (x.depth + 1) with CAS.

        NOTE: The caller is responsible for fusing gamma and beta into the
        next matmul's weights (v106 fusion rule).
        """
        if x._cipher is None:
            # Plain mode (training) — dispatch to forward_plain
            token_regions_t = None
            if token_regions is not None:
                token_regions_t = torch.tensor(token_regions, dtype=torch.long)
            return CiphertextTensor(
                data=self.forward_plain(x._plain, token_regions_t),
                engine=x._engine,
                depth=x.depth,
            )

        # Cipher mode (inference)
        engine = x._engine
        ciphers = x._cipher  # list of TenSEAL CKKSVectors

        if x_sq is not None:
            # v152 CAS: Use pre-computed x² (cipher×plain ONLY, NO cipher×cipher)
            # z = c0 + c1·x + c2·x_sq
            # All multiplications are cipher×scalar (+1 depth), additions are +0.
            # Total depth: +1 (was +2 without CAS). NO correlated noise.
            sq_ciphers = x_sq._cipher
            out_ciphers = []
            if self.num_regions == 1 or token_regions is None:
                c0 = self._cipher_c0 if self._cipher_c0 is not None else float(self.c0.mean())
                c1 = self._cipher_c1 if self._cipher_c1 is not None else float(self.c1.mean())
                c2 = self._cipher_c2 if self._cipher_c2 is not None else float(self.c2.mean())
                for ct, ct_sq in zip(ciphers, sq_ciphers):
                    # c1·x (cipher×scalar, +1 depth)
                    term1 = ct * c1
                    # c2·x_sq (cipher×scalar, +1 depth — x_sq is fresh, depth 0)
                    term2 = ct_sq * c2
                    # c0 + term1 + term2 (cipher+cipher+scalar, +0 depth)
                    z_norm = term1 + term2 + c0
                    out_ciphers.append(z_norm)
            else:
                # v117 per-region coeffs with CAS
                for i, (ct, ct_sq) in enumerate(zip(ciphers, sq_ciphers)):
                    r = token_regions[i]
                    c0 = self._cipher_c0_regions[r]
                    c1 = self._cipher_c1_regions[r]
                    c2 = self._cipher_c2_regions[r]
                    term1 = ct * c1
                    term2 = ct_sq * c2
                    z_norm = term1 + term2 + c0
                    out_ciphers.append(z_norm)

            return CiphertextTensor(
                ciphertexts=out_ciphers,
                engine=engine,
                depth=x.depth + 1,  # CAS: +1 (was +2)
            )

        # Original path (no CAS): polyval with cipher×cipher (depth +2)
        if self.poly_degree == 1:
            out_ciphers = []
            if self.num_regions == 1 or token_regions is None:
                c0 = self._cipher_c0 if self._cipher_c0 is not None else float(self.c0.mean())
                c1 = self._cipher_c1 if self._cipher_c1 is not None else float(self.c1.mean())
                for ct in ciphers:
                    out_ciphers.append(ct * c1 + c0)
            else:
                for i, ct in enumerate(ciphers):
                    r = token_regions[i]
                    out_ciphers.append(
                        ct * self._cipher_c1_regions[r] + self._cipher_c0_regions[r]
                    )
            return CiphertextTensor(
                ciphertexts=out_ciphers,
                engine=engine,
                depth=x.depth + 1,
            )

        out_ciphers = []
        if self.num_regions == 1 or token_regions is None:
            # v116 path: SHARED coeffs (mean across D)
            c0 = self._cipher_c0 if self._cipher_c0 is not None else float(self.c0.mean())
            c1 = self._cipher_c1 if self._cipher_c1 is not None else float(self.c1.mean())
            c2 = self._cipher_c2 if self._cipher_c2 is not None else float(self.c2.mean())
            coeffs = [c0, c1, c2]
            for ct in ciphers:
                z_norm = ct.polyval(coeffs)
                out_ciphers.append(z_norm)
        else:
            # v117 path: PER-REGION coeffs (each token uses its region's coeffs)
            # token_regions: list of N region indices (0..R-1)
            # _cipher_c0_regions: list of R scalars
            for i, ct in enumerate(ciphers):
                r = token_regions[i]
                coeffs = [
                    self._cipher_c0_regions[r],
                    self._cipher_c1_regions[r],
                    self._cipher_c2_regions[r],
                ]
                if self.channelwise_coeffs:
                    # TenSEAL polyval accepts scalar coefficients only.  The
                    # channelwise path evaluates the same quadratic with
                    # vector plaintext coefficients, preserving two levels.
                    square = ct * ct
                    z_norm = square * coeffs[2] + ct * coeffs[1] + coeffs[0]
                else:
                    z_norm = ct.polyval(coeffs)
                out_ciphers.append(z_norm)

        return CiphertextTensor(
            ciphertexts=out_ciphers,
            engine=engine,
            depth=x.depth + self.poly_degree,
        )

    def prepare_cipher_weights(self):
        """Cache polynomial coefficients for cipher path.

        v116: Cache SHARED coeffs (mean across D) as scalars.
        v117: Cache PER-REGION coeffs as lists of R scalars.

        The gamma/beta affine is NOT applied here — the caller must fuse
        them into the next matmul's weights.
        """
        if self.num_regions == 1:
            # v116 path: shared coeffs (mean across D)
            self._cipher_c0 = float(self.c0.detach().mean())
            self._cipher_c1 = float(self.c1.detach().mean())
            self._cipher_c2 = float(self.c2.detach().mean())
        else:
            # v117 path: per-region coeffs. Channelwise coefficients are kept
            # as lists so TenSEAL applies one plaintext value per CKKS slot.
            self._cipher_c0_regions = self.c0_regions.detach().cpu().tolist()
            self._cipher_c1_regions = self.c1_regions.detach().cpu().tolist()
            self._cipher_c2_regions = self.c2_regions.detach().cpu().tolist()

    def multiplicative_depth(self) -> int:
        """Return FHE multiplicative depth of PolyNorm.

        v116 and v117 both: poly_degree (2 for degree-2)
        Per-region coeffs do NOT add depth (same polyval operation).
        """
        return self.poly_degree
