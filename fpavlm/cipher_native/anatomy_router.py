"""AnatomyAwareTokenRouter: Medical Prior-Based Token Region Assignment for FHE ViT.

v117 INNOVATION: Anatomy-Aware Polynomial Normalization (AAPN) enabler.

  v116 limitation: AnatomyRouter cipher path was IDENTITY — the "anatomy-aware"
    claim was hollow in FHE mode. The router existed for plain mode only.

  v117 solution: AnatomyRouter becomes a CONFIGURATION provider that computes
    FIXED token_regions (based on 2D grid position). The actual region-specific
    processing happens in PolyNorm (per-region polynomial coefficients).

  Architecture:
    AnatomyRouter.compute_token_regions() → list of N region indices (0..R-1)
    Encoder passes token_regions to each MedFHEBlock
    MedFHEBlock passes token_regions to PolyNorm
    PolyNorm applies per-region polyval coeffs (ZERO extra FHE depth)

  Why this is FHE-native:
    - Region assignment is FIXED (based on position, not data) — no conditional branch
    - No rotate needed (regions are per-token, each token is separate cipher)
    - No cipher×plain-vector (per-region coeffs are scalars applied via polyval)
    - ZERO extra FHE depth (same polyval operation, just different coeffs)

  Medical prior (chest X-ray, R=5 regions):
    - Region 0: Upper lung fields (apices) — pneumothorax, tuberculosis
    - Region 1: Middle lung L (hila, bronchovascular) — hilar adenopathy
    - Region 2: Cardiac silhouette (center/right) — cardiomegaly, heart failure
    - Region 3: Lower lung L (costophrenic) — effusion, consolidation
    - Region 4: Lower lung R / mediastinum — mediastinal mass, rib fractures

  v155 SYMMETRY PRIOR (medical-specific): Chest X-rays are approximately
  left-right symmetric. Pathology often breaks this symmetry (unilateral
  effusion, unilateral pneumothorax). The symmetry map defines contralateral
  region pairs that should share PolyNorm coefficients under the symmetry
  constraint:
    - Region 1 (mid lung L) ↔ Region 4 (lower lung R / mediastinum)  [NOT exact
      mirror due to cardiac asymmetry, but closest contralateral pair]
    - Region 3 (lower lung L) ↔ Region 4 (lower lung R)
    - Region 0 (upper lung, bilateral) and Region 2 (cardiac, central) are
      self-symmetric (no contralateral pair).
  This is a medical prior, not a generic symmetry assumption — it encodes
  the radiological knowledge that bilateral comparison is diagnostic.

  Privacy:
    - Region assignment is a FIXED lookup (no model parameters leaked)
    - Per-region polynomial coefficients are SERVER-side model parameters
    - Client never sees coefficients; server never sees plaintext tokens

  The router contains no learnable parameters; its only role is to provide the
  fixed token-to-region assignment.
"""
from __future__ import annotations

from typing import List, Optional

import torch.nn as nn


class AnatomyAwareTokenRouter(nn.Module):
    """Medical prior-based token region assignment for FHE ViT (v117+).

    v117 role: Provides FIXED token_regions (list of N region indices) based on
    2D grid position. The actual region-specific processing happens in PolyNorm
    via per-region polynomial coefficients (AAPN).

    The router has NO learnable parameters in v117+ — it is a pure configuration
    provider. The region assignment is deterministic (based on 2D grid position),
    so it adds zero FHE depth and zero privacy surface.

    Medical prior: 5 anatomical regions for chest X-ray (configurable).
    """

    def __init__(
        self,
        dim: int,
        num_tokens: int = 16,
        grid_size: int = 4,  # sqrt(num_tokens) for 2D position
        num_regions: int = 5,
        spectral_bound: float = 3.0,  # kept for API compat, unused
    ):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.grid_size = grid_size
        self.num_regions = num_regions
        # Cached token_regions (computed once, FIXED)
        self._token_regions: Optional[List[int]] = None

    def _token_to_region(self, token_idx: int) -> int:
        """Map a token index to its anatomical region (FIXED, based on 2D position).

        Args:
            token_idx: token index in [0, num_tokens)

        Returns:
            region index in [0, num_regions)

        Region assignment:
          - R=5 (chest X-ray medical prior):
            - row_frac < 0.33: region 0 (upper lung)
            - row_frac < 0.67 and col < grid_size//2: region 1 (middle lung L)
            - row_frac < 0.67 and col >= grid_size//2: region 2 (cardiac)
            - row_frac >= 0.67 and col < grid_size//2: region 3 (lower lung L)
            - row_frac >= 0.67 and col >= grid_size//2: region 4 (lower lung R / mediastinum)
          - R≠5 (ablation): uniform partition of token_idx into R regions
            This provides a fair comparison for the per-region ablation
            (tests whether per-region coeffs help, regardless of specific assignment)
        """
        # v121: For R≠5, use uniform partition (ablation mode)
        if self.num_regions != 5:
            return token_idx * self.num_regions // self.num_tokens

        # R=5: medical prior (chest X-ray anatomical regions)
        row = token_idx // self.grid_size  # 0..grid_size-1
        row_frac = row / max(self.grid_size - 1, 1)  # 0..1
        col = token_idx % self.grid_size

        if row_frac < 0.33:
            return 0  # upper lung
        elif row_frac < 0.67:
            if col < self.grid_size // 2:
                return 1  # middle lung L
            else:
                return 2  # cardiac
        else:
            if col < self.grid_size // 2:
                return 3  # lower lung L
            else:
                return 4  # lower lung R / mediastinum

    def compute_token_regions(self) -> List[int]:
        """Compute FIXED token_regions list based on 2D grid position.

        Returns:
            list of N region indices (0..R-1), one per token

        This is the v117 key method: provides the region assignment that
        PolyNorm uses to apply per-region polynomial coefficients.
        The assignment is FIXED (anatomical prior), NOT data-dependent.
        """
        if self._token_regions is None:
            self._token_regions = [self._token_to_region(i) for i in range(self.num_tokens)]
        return self._token_regions

    def get_symmetry_pairs(self) -> List[tuple]:
        """Return contralateral region pairs for the v155 symmetry prior.

        Each tuple (r_a, r_b) means regions r_a and r_b are contralateral
        (left-right mirror) and should share PolyNorm coefficients under the
        symmetry constraint. Self-symmetric regions (upper lung, cardiac) are
        NOT included — they have no contralateral pair.

        Returns:
            List of (r_a, r_b) pairs. For R=5 chest X-ray:
              [(1, 4), (3, 4)] — mid-lung-L↔lower-R, lower-L↔lower-R.
            For R≠5: empty list (symmetry prior is chest-X-ray-specific).
        """
        if self.num_regions != 5:
            return []
        # v155 medical symmetry: left lung regions (1, 3) pair with right-side
        # region (4). Region 0 (bilateral upper lung) and 2 (central cardiac)
        # are self-symmetric.
        return [(1, 4), (3, 4)]

    def multiplicative_depth(self) -> int:
        """Return FHE depth of router (0 — identity, regions applied in PolyNorm)."""
        return 0

    def bake_statistics(self):
        """Bake statistics for cipher path (no-op in v117+ — router has no params)."""
        self.compute_token_regions()
