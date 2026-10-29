"""Position-Based Selective Refresh (PBSR) for Medical FHE-ViT.

MOTIVATION — MEDICAL IMAGE STRUCTURE EXPLOITED FOR FHE EFFICIENCY:

  In chest X-ray (CXR) and other medical images, the distribution of
  diagnostic information is highly NON-UNIFORM across image regions:

    [corner]  [top_L]   [top_R]   [corner]
    [mid_L]   [lung_L]  [lung_R]  [mid_R]
    [mid_L]   [cardiac] [cardiac] [mid_R]
    [corner]  [low_L]   [low_R]   [corner]

  This fixed-coordinate heuristic prioritizes central lung and cardiac regions.
  It does not establish where all clinically relevant findings occur.
  CORNER patches (4 of 16) may contain:
    - Ribs (repetitive texture, low diagnostic variance)
    - Image background / scanner borders (near-zero signal)
    - Lateral soft tissue (low diagnostic relevance)

  PolyNorm approximation error on these patches must be measured; a spatial
  coordinate prior does not guarantee small or near-Gaussian activations.

  PBSR exploits this structure to REDUCE REFRESH COST:
    - Diagnostic patches (lung fields, cardiac): CADN refresh (exact LayerNorm)
    - Background patches (corners, periphery): PolyNorm (no refresh needed)
    - Per-refresh cost reduction: ~25-40% fewer decrypt/re-encrypt operations

PBSR ALGORITHM:
  1. Before FHE inference, compute a static "diagnostic mask" M ∈ {0,1}^N
     using the AnatomyAwareTokenRouter.  M is IMAGE-INDEPENDENT (uses only
     spatial position and anatomy priors, NOT patient data).
  2. At each CADN refresh boundary:
     - For masked tokens (M=1): CADN refresh → exact LayerNorm
     - For unmasked tokens (M=0): PolyNorm fallback → polynomial approx
  3. Merge the refreshed and PolyNorm-processed tokens for the next block.

PRIVACY PRESERVATION:
  The PBSR mask M is IMAGE-AGNOSTIC: it is computed from the FIXED anatomy
  prior (spatial grid position), not from the encrypted patient data.
  The server cannot infer patient information from M.  The mask is
  transparently published as part of the model configuration.

COMPOSITIONAL ANALYSIS:
  PBSR composes with LBR (LazyBlockRefreshScheduler):
    - LBR decides WHICH BLOCKS get CADN
    - PBSR decides WHICH TOKENS within those blocks get CADN
  Combined, they reduce total CADN ops from 5×16=80 to ~2×10=20 (75% ↓).

ACCURACY MODEL:
  For background tokens (M=0 in PBSR), PolyNorm error ε_poly propagates
  through subsequent blocks.  The error is bounded by:

      ||z_poly - z_cadn||_2 ≤ C_poly × ||x||_2^2 × (1 - Taylor accuracy)

  This is an informal historical model, not a proved bound for the deployed
  encoder. Small background error or preserved diagnosis is not guaranteed.

RETAINED INTERACTIVE BASELINE (not a literature-first claim):
  1. Per-token selective client refresh in the inherited FHE-ViT pipeline.
  2. IMAGE-AGNOSTIC: uses spatial anatomy priors, no per-image decryption.
  3. Composable with LBR; elapsed speedup requires measurement.
  4. Medical-specific: exploits the structured sparsity of CXR findings.
  5. Zero training change: purely an inference-time optimization.

LATENCY MODEL (3 blocks, 16 tokens, CADN+PBSR combined):
  Full CADN: 5 round trips × 16 tokens = 80 token-refreshes
  LBR only:  2 round trips × 16 tokens = 32 token-refreshes (2.5× ↓)
  PBSR only: 5 round trips × 10 tokens = 50 token-refreshes (1.6× ↓)
  LBR+PBSR:  2 round trips × 10 tokens = 20 token-refreshes (4× ↓)

These are operation-count estimates, not measured 0.4 codec latency. CADN is
client decryption/re-encryption, not homomorphic bootstrapping. Refer to
AnatomyAwareTokenRouter for the coordinate prior and RELATED_WORK.md for verified
prior art. This scheduler is not on the clinical codec's callback-free path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# Standard CXR anatomy regions for a 4×4 patch grid (patch index 0–15,
# row-major: row 0 = top, col 0 = left).
# Based on standard AP/PA CXR anatomy:
#   Upper zones: row 0 (patches 0–3)
#   Mid/cardiac zones: rows 1–2 (patches 4–11)
#   Lower zones: row 3 (patches 12–15)
#   Lung fields: columns 0–1 (left) and 2–3 (right)
#   Cardiac: center of rows 1–2

# Default diagnostic patch set for 4×4 grid (16 patches):
#   Diagnostic (lung fields + cardiac silhouette + perihilar):
#     patches 1,2,4,5,6,7,8,9,10,11  (10 of 16 = 62.5%)
#   Background (corners + top periphery + lower periphery):
#     patches 0,3,12,13,14,15  (6 of 16 = 37.5%)
_DEFAULT_DIAGNOSTIC_PATCHES_4x4 = frozenset([1, 2, 4, 5, 6, 7, 8, 9, 10, 11])
_DEFAULT_BACKGROUND_PATCHES_4x4 = frozenset([0, 3, 12, 13, 14, 15])


@dataclass(frozen=True)
class PBSRMask:
    """Static per-token diagnostic mask for PBSR.

    Attributes:
        diagnostic_indices: Indices of tokens that should receive CADN refresh.
        background_indices: Indices of tokens that fall back to PolyNorm.
        num_tokens: Total number of tokens.
        grid_size: Side length of the spatial grid (num_tokens = grid_size²).
        coverage_rate: Fraction of tokens in the diagnostic set.
    """
    diagnostic_indices: Tuple[int, ...]
    background_indices: Tuple[int, ...]
    num_tokens: int
    grid_size: int
    coverage_rate: float

    def as_bool_tensor(self) -> torch.Tensor:
        """Return [num_tokens] bool tensor: True for diagnostic tokens."""
        mask = torch.zeros(self.num_tokens, dtype=torch.bool)
        for i in self.diagnostic_indices:
            mask[i] = True
        return mask

    def __repr__(self) -> str:
        return (
            f"PBSRMask(diagnostic={len(self.diagnostic_indices)}/"
            f"{self.num_tokens}, coverage={self.coverage_rate:.1%})"
        )


class PBSRScheduler:
    """Compute and apply Position-Based Selective Refresh masks.

    Usage::
        scheduler = PBSRScheduler(num_tokens=16, grid_size=4)
        mask = scheduler.compute_mask()
        # mask.diagnostic_indices → tokens that need CADN refresh
        # mask.background_indices → tokens that use PolyNorm

        # Apply during CADN callback to select which ciphertexts to refresh
        all_cts = [ct_0, ct_1, ..., ct_15]
        refreshed = scheduler.selective_refresh(all_cts, mask, cadn_fn, polynorm_fn)
    """

    def __init__(
        self,
        num_tokens: int = 16,
        grid_size: int = 4,
        anatomy_regions: Optional[Dict[str, List[int]]] = None,
        diagnostic_region_names: Optional[Sequence[str]] = None,
        coverage_target: float = 0.6,
    ):
        """
        Args:
            num_tokens: Total number of image patches (e.g. 16 for a 4×4 grid).
            grid_size: Side length of the spatial patch grid.
            anatomy_regions: Optional dict mapping region name → list of patch
                indices (e.g. from AnatomyAwareTokenRouter).  When provided,
                ``diagnostic_region_names`` selects which regions are diagnostic.
            diagnostic_region_names: Region names to mark as diagnostic.  Only
                used when ``anatomy_regions`` is provided.
            coverage_target: If anatomy_regions is not provided, the scheduler
                selects the innermost ``coverage_target`` fraction of patches
                as diagnostic (based on distance from image centre).
        """
        if num_tokens < 1:
            raise ValueError("num_tokens must be ≥ 1")
        if grid_size * grid_size != num_tokens:
            raise ValueError(
                f"grid_size² ({grid_size}² = {grid_size**2}) must equal "
                f"num_tokens ({num_tokens})"
            )
        if not 0.0 < coverage_target <= 1.0:
            raise ValueError("coverage_target must be in (0, 1]")

        self.num_tokens = num_tokens
        self.grid_size = grid_size
        self.anatomy_regions = anatomy_regions
        self.diagnostic_region_names = diagnostic_region_names
        self.coverage_target = coverage_target

    def compute_mask(self) -> PBSRMask:
        """Compute the static per-token diagnostic mask.

        Returns:
            PBSRMask with diagnostic and background patch indices.
        """
        if self.anatomy_regions is not None and self.diagnostic_region_names is not None:
            # Use provided anatomy regions
            diag_set = set()
            for name in self.diagnostic_region_names:
                if name in self.anatomy_regions:
                    diag_set.update(self.anatomy_regions[name])
        elif self.num_tokens == 16 and self.grid_size == 4:
            # Use default CXR 4×4 anatomy map
            diag_set = set(_DEFAULT_DIAGNOSTIC_PATCHES_4x4)
        else:
            # Fallback: select innermost patches by distance from centre
            diag_set = self._select_by_centre_proximity()

        # Validate indices
        diag_set = {i for i in diag_set if 0 <= i < self.num_tokens}
        bg_set = set(range(self.num_tokens)) - diag_set

        coverage = len(diag_set) / self.num_tokens
        return PBSRMask(
            diagnostic_indices=tuple(sorted(diag_set)),
            background_indices=tuple(sorted(bg_set)),
            num_tokens=self.num_tokens,
            grid_size=self.grid_size,
            coverage_rate=coverage,
        )

    def _select_by_centre_proximity(self) -> set:
        """Select patches closest to image centre as diagnostic."""
        n_diag = max(1, round(self.num_tokens * self.coverage_target))
        centre = (self.grid_size - 1) / 2.0
        distances = []
        for idx in range(self.num_tokens):
            row = idx // self.grid_size
            col = idx % self.grid_size
            dist = (row - centre) ** 2 + (col - centre) ** 2
            distances.append((dist, idx))
        distances.sort()
        return {idx for _, idx in distances[:n_diag]}

    def selective_refresh(
        self,
        ciphertexts: List,
        mask: PBSRMask,
        cadn_refresh_fn,
        polynorm_values: Optional[List] = None,
    ) -> List:
        """Apply PBSR: selective CADN for diagnostic tokens, PolyNorm for background.

        Args:
            ciphertexts: List of N TenSEAL CKKS ciphertexts (one per token).
            mask: PBSRMask from ``compute_mask()``.
            cadn_refresh_fn: Callable that takes a LIST of ciphertexts and
                returns (fresh_cts, z_norm_cts); called only for diagnostic
                tokens.
            polynorm_values: Optional pre-computed PolyNorm outputs for
                background tokens (None to use baked statistics inside the
                encoder block).

        Returns:
            List of N ciphertexts after selective refresh.  Diagnostic tokens
            are freshly re-encrypted; background tokens retain their current
            ciphertext (to be processed by PolyNorm in the next block).
        """
        result = list(ciphertexts)  # shallow copy

        # Extract diagnostic ciphertexts
        diag_cts = [ciphertexts[i] for i in mask.diagnostic_indices]

        # CADN refresh diagnostic tokens
        fresh_diag_cts, _ = cadn_refresh_fn(diag_cts)

        # Put refreshed tokens back
        for new_idx, orig_idx in enumerate(mask.diagnostic_indices):
            result[orig_idx] = fresh_diag_cts[new_idx]

        # Background tokens: keep current ciphertext (PolyNorm will handle them)
        # They do NOT get a depth reset, so the encoder block must use PolyNorm
        # for these tokens specifically.
        return result

    def expected_speedup(
        self,
        lbr_n_refreshes: int,
        round_trip_cost_per_token: float = 1.0,
    ) -> float:
        """Estimate throughput speedup vs full CADN (all tokens, all blocks).

        Args:
            lbr_n_refreshes: Number of block-level CADN refreshes (from LBR).
            round_trip_cost_per_token: Cost of refreshing one token (normalised).

        Returns:
            Speedup factor (> 1.0 means faster than full CADN).
        """
        mask = self.compute_mask()
        n_full = 5 * self.num_tokens  # 5 refreshes × 16 tokens = 80
        n_pbsr_lbr = lbr_n_refreshes * len(mask.diagnostic_indices)
        return n_full / max(n_pbsr_lbr, 1)

    @staticmethod
    def from_anatomy_router(router) -> "PBSRScheduler":
        """Construct a PBSRScheduler from an AnatomyAwareTokenRouter instance.

        The router's region assignments are used to determine which patches
        belong to diagnostically relevant regions.

        Args:
            router: AnatomyAwareTokenRouter instance.

        Returns:
            PBSRScheduler configured with the router's region map.
        """
        # Get region assignments: {region_idx: [patch_indices]}
        region_assignments = {}
        for patch_idx in range(router.num_tokens):
            region_idx = int(router.get_region(patch_idx))
            region_assignments.setdefault(region_idx, []).append(patch_idx)

        # Default: all non-corner regions are diagnostic
        # Corner regions = regions where all patches are at grid corners
        g = router.grid_size
        corner_patches = {0, g - 1, g * (g - 1), g * g - 1}
        diagnostic_regions = []
        for region_idx, patches in region_assignments.items():
            if not set(patches).issubset(corner_patches):
                diagnostic_regions.append(str(region_idx))

        return PBSRScheduler(
            num_tokens=router.num_tokens,
            grid_size=router.grid_size,
            anatomy_regions={str(k): v for k, v in region_assignments.items()},
            diagnostic_region_names=diagnostic_regions,
        )


def compute_pbsr_lbr_speedup(
    num_blocks: int,
    num_tokens: int,
    pbsr_coverage: float,
    lbr_cadn_fraction: float,
    round_trip_cost_ratio: float = 4.0,
) -> Dict[str, float]:
    """Compute speedup metrics for the combined PBSR + LBR optimization.

    Args:
        num_blocks: Number of transformer blocks.
        num_tokens: Total number of image tokens.
        pbsr_coverage: Fraction of tokens refreshed per PBSR (e.g. 0.625 for 10/16).
        lbr_cadn_fraction: Fraction of blocks using CADN (from LBR).
        round_trip_cost_ratio: Ratio of round-trip to arithmetic cost.

    Returns:
        Dict with speedup metrics for PBSR-only, LBR-only, and combined.
    """
    n_refreshes_full = num_blocks + 2  # standard CADN
    n_tokens_full = num_tokens

    n_refreshes_lbr = max(1, round(n_refreshes_full * lbr_cadn_fraction))
    n_tokens_pbsr = round(num_tokens * pbsr_coverage)

    def cost(n_ref, n_tok):
        return n_ref * round_trip_cost_ratio * n_tok + num_blocks

    cost_full = cost(n_refreshes_full, n_tokens_full)
    cost_pbsr_only = cost(n_refreshes_full, n_tokens_pbsr)
    cost_lbr_only = cost(n_refreshes_lbr, n_tokens_full)
    cost_combined = cost(n_refreshes_lbr, n_tokens_pbsr)

    return {
        "speedup_pbsr_only": cost_full / max(cost_pbsr_only, 1e-6),
        "speedup_lbr_only": cost_full / max(cost_lbr_only, 1e-6),
        "speedup_combined": cost_full / max(cost_combined, 1e-6),
        "n_refreshes_full": n_refreshes_full,
        "n_refreshes_lbr": n_refreshes_lbr,
        "n_tokens_full": n_tokens_full,
        "n_tokens_pbsr": n_tokens_pbsr,
        "refresh_ops_full": n_refreshes_full * n_tokens_full,
        "refresh_ops_combined": n_refreshes_lbr * n_tokens_pbsr,
        "refresh_ops_reduction_pct": float(
            100.0
            * (1.0 - (n_refreshes_lbr * n_tokens_pbsr) / (n_refreshes_full * n_tokens_full))
        ),
    }


__all__ = [
    "PBSRMask",
    "PBSRScheduler",
    "compute_pbsr_lbr_speedup",
]
