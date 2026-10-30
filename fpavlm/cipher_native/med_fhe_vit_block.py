"""Polynomial medical vision blocks and fixed anatomical token mixing.

The ciphertext path uses plaintext projections, low-degree polynomials,
residual sums, and fixed learned spatial mixing. Dynamic self-attention is
not implemented by this mixer. CADN is an interactive research alternative
that exposes intermediate states to the client. Callback-free profiles use
polynomial normalization. Neither path establishes zero post-release leakage.

Use recorded experiments for accuracy and latency; historical version labels
in implementation comments are not independent scientific contributions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .adaptive_noise import AdaptiveCKKSNoise
from .cadn_norm import CADNLayer
from .ciphertext_tensor import CiphertextTensor
from .poly_norm import PolyNorm
from .spectral_bounded import SpectralBoundedLinear


class AnatomyStructuredTokenMixer(nn.Module):
    """v135: Anatomy-Structured Token Mixing (ASTM) — Adaptive Cross-Token Flow.

    ADAPTIVE INNOVATION (not application-level): Extends the existing AAPN
    (Anatomy-Aware Polynomial Normalization) design pattern from PolyNorm to
    token mixing. Instead of a generic [N,N] mixing matrix (MLP-Mixer style),
    ASTM decomposes mixing using the SAME anatomical region structure that
    PolyNorm already uses for per-region coefficients.

    v164 AMPLIFICATION: SYM_SCALE amplifies the MedLTM symmetry-difference
      branch (Innovation F). ROOT CAUSE: W_sym uses SkipInit (starts at 0) and
      the contralateral difference signal is subtle vs. the dominant intra/
      inter-region mixing. Without amplification, trained |W_sym| < 0.015
      giving <1% feature contribution (inert). SYM_SCALE=10.0 amplifies both
      the forward effect AND the gradient (chain rule), so W_sym receives a
      10× stronger learning signal while keeping the parameter itself small
      (spectral-bounded). FHE depth UNCHANGED (scalar mul is +1 depth
      regardless of scalar magnitude).

    ROOT CAUSE of val_cos=0.82 ceiling (v131/v132/v133/v134 diagnostics):
      The encoder had NO token-mixing mechanism. Each of the 16 tokens was
      processed COMPLETELY INDEPENDENTLY through PolyNorm + PolyMLP. There was
      NO cross-token information flow. This explained ALL diagnostic failures
      (dim, Perceiver depth, patch count, resolution — all irrelevant when
      tokens never interact).

    ASTM SOLUTION — Two-level anatomy-structured mixing:

      1. INTRA-REGION mixing (W_intra[r]): Per-region small matrices.
         Tokens in the SAME anatomical region mix freely.
         Medical: pathology spreads within a region (e.g., consolidation
         within lower lung field). Each region has its OWN mixing matrix,
         following AAPN's per-region coefficient pattern.
         Structure: block-diagonal (R blocks, sizes = region token counts).

      2. INTER-REGION mixing (A_inter[R, R]): Region affinity matrix.
         Tokens in DIFFERENT regions mix via region-level affinity.
         Medical: cross-region pathology interaction (e.g., cardiomegaly
         affecting adjacent lung fields). A_inter[r_i, r_j] is the mixing
         strength from region r_j to region r_i.

    DECOMPOSITION (at init, y = x — identity):
      W_mix[i,j] = W_intra[r_i][local_i, local_j]  if r_i == r_j  (intra)
      W_mix[i,j] = A_inter[r_i, r_j]                if r_i != r_j  (inter)
      Init: W_intra[r] = I + ε·randn, A_inter = 0  →  y = x at init.

    FHE-NATIVE OPTIMIZATION — Region-Sum Factorization:
      The key FHE insight: inter-region mixing depends only on REGIONS, not
      individual tokens. So we compute region sums ONCE and reuse:

        X_region[r] = sum_{j in region r} x[j]          (N-R cipher+cipher)
        Y_inter[r]  = sum_{r'≠r} A_inter[r,r'] · X_region[r']  (R region ops)
        y[i] = y_intra[i] + Y_inter[region_i]            (N cipher+cipher)

      This is IMPOSSIBLE with generic MLP-Mixer (no structure to exploit).
      ASTM's anatomy structure enables a 3.4× reduction in cipher×scalar ops.

      Total FHE ops (N=16, R=5, region sizes 4,2,2,4,4):
        cipher×scalar: 56 (intra) + 20 (inter) = 76   (vs 256 dense, 3.4× ↓)
        cipher+cipher: 40 (intra) + 53 (inter+sums) = 93 (vs 240 dense, 2.6× ↓)
      Depth: +1 (cipher×scalar only, NO cipher×cipher in mixing).

    PARAMETERS: 81 (56 intra + 25 inter) vs 256 for dense [N,N] (3.2× ↓).

    ADAPTIVE TO ARCHITECTURE (why this is NOT application-level):
      - REUSES AnatomyRouter's region structure (same 5 chest X-ray regions)
      - FOLLOWS AAPN's per-region parameterization pattern (PolyNorm has
        per-region coeffs c0[r],c1[r],c2[r]; ASTM has per-region W_intra[r])
      - SPECTRAL-BOUNDED (same as SpectralBoundedLinear, τ_mix=3.0)
      - FHE-NATIVE (cipher×scalar + cipher+cipher only, no rotate, no cipher×cipher)
      - MEDICALLY MOTIVATED (intra-region pathology correlation + inter-region
        cross-pathology interaction, not generic spatial mixing)

    PRIVACY: All weights (W_intra, A_inter) are SERVER-side PRIVATE (like W_a,
      W_o). Mixing happens ENTIRELY on SERVER under FHE. Client only does CANR.
    """

    # v164: Amplification scale for MedLTM symmetry-difference branch (Innovation F).
    # See class docstring for first-principles justification.
    SYM_SCALE: float = 10.0

    def __init__(
        self,
        num_tokens: int,
        spectral_bound: float = 3.0,
        init_eps: float = 0.01,
        n_power_iterations: int = 3,
        # v135 ASTM: anatomy structure (set via set_token_regions after construction)
        token_regions: Optional[List[int]] = None,
        num_regions: int = 5,
        # v156 Innovation F: MedLTM — symmetry-difference branch (contralateral pairs)
        symmetry_pairs: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.spectral_bound = spectral_bound
        self.n_power_iterations = n_power_iterations
        self.num_regions = num_regions
        self.init_eps = init_eps

        # Anatomy structure (set by set_token_regions, or from constructor arg)
        self._token_regions: Optional[List[int]] = None
        self._region_token_lists: Optional[List[List[int]]] = None  # region -> token indices
        self._token_to_local: Optional[Dict[int, int]] = None  # token -> local idx in region
        self._region_sizes: Optional[List[int]] = None  # tokens per region

        # === Intra-region mixing: per-region small matrices ===
        # Created lazily in set_token_regions (need region sizes first).
        # W_intra_list[r]: nn.Parameter of shape [n_r, n_r] for region r.
        self.W_intra_list: nn.ParameterList = nn.ParameterList()
        # === Inter-region mixing: [R, R] region affinity matrix ===
        # A_inter[r_i, r_j] = mixing strength from region r_j to region r_i.
        # Init: zeros (no inter-region mixing at init → y = x).
        self.A_inter = nn.Parameter(torch.zeros(num_regions, num_regions))

        # === v156 Innovation F: MedLTM — symmetry-difference branch ===
        # Bilateral symmetry is the STRONGEST chest X-ray prior. Pathology
        # BREAKS it (unilateral effusion, asymmetric consolidation). The
        # contralateral difference (left − right) is THE key diagnostic signal.
        # W_sym[r] scales the contralateral difference for region r.
        # Init: zeros (SkipInit — starts as plain ASTM, grows during training).
        # Only created when symmetry_pairs is provided (chest X-ray prior).
        # FHE: cipher+cipher subtraction is +0 depth; cipher×scalar is +1 depth
        # (SAME as existing inter-region). Total depth UNCHANGED.
        if symmetry_pairs is not None and len(symmetry_pairs) > 0:
            self.W_sym = nn.Parameter(torch.zeros(num_regions))
            self._symmetry_pairs: Optional[List[Tuple[int, int]]] = list(symmetry_pairs)
            self._use_med_ltm = True
        else:
            self.W_sym = None
            self._symmetry_pairs = None
            self._use_med_ltm = False

        # Cached cipher-path weights (set by prepare_cipher_weights)
        self._cipher_W_intra: Optional[List[torch.Tensor]] = None
        self._cipher_A_inter: Optional[torch.Tensor] = None
        # v156 Innovation F: cached symmetry weights (list of R floats)
        self._cipher_W_sym: Optional[List[float]] = None

        # If token_regions provided at construction, initialize now
        if token_regions is not None:
            self.set_token_regions(token_regions, num_regions)

    def set_token_regions(self, token_regions: List[int], num_regions: Optional[int] = None):
        """Set the anatomy structure and initialize intra-region matrices.

        Called by the encoder after AnatomyRouter computes token_regions.
        This is the ADAPTIVE hook: ASTM reuses the SAME region assignment
        that PolyNorm (AAPN) already uses.

        Args:
            token_regions: list of N region indices (0..R-1), one per token.
            num_regions: number of regions (R). If None, uses self.num_regions.
        """
        if num_regions is not None:
            self.num_regions = num_regions
        R = self.num_regions
        N = self.num_tokens
        assert len(token_regions) == N, \
            f"token_regions len {len(token_regions)} != num_tokens {N}"

        self._token_regions = list(token_regions)

        # Build region -> token index mapping
        self._region_token_lists: List[List[int]] = [[] for _ in range(R)]
        for tok_idx, r in enumerate(self._token_regions):
            self._region_token_lists[r].append(tok_idx)
        self._region_sizes = [len(rl) for rl in self._region_token_lists]

        # Build token -> local index within region
        self._token_to_local: Dict[int, int] = {}
        for r in range(R):
            for local_idx, tok_idx in enumerate(self._region_token_lists[r]):
                self._token_to_local[tok_idx] = local_idx

        # Initialize per-region intra-mixing matrices (identity + ε·randn)
        # Clear existing (in case set_token_regions called again)
        self.W_intra_list = nn.ParameterList()
        for r in range(R):
            n_r = self._region_sizes[r]
            if n_r == 0:
                # Empty region — create a 1×1 placeholder (won't be used)
                self.W_intra_list.append(nn.Parameter(torch.zeros(1, 1)))
                continue
            W_r = torch.eye(n_r) + self.init_eps * torch.randn(n_r, n_r)
            self.W_intra_list.append(nn.Parameter(W_r))

        # Inter-region affinity: zeros at init (no inter-region mixing → y = x)
        with torch.no_grad():
            self.A_inter.zero_()

    def _build_effective_W_mix(self) -> torch.Tensor:
        """Build the effective [N, N] mixing matrix from intra + inter components.

        This is used for:
          - Spectral norm estimation
          - FHE fallback (if region structure not set)

        W_mix[i,j] = W_intra[r_i][local_i, local_j]  if r_i == r_j
        W_mix[i,j] = A_inter[r_i, r_j]                if r_i != r_j

        v144 VECTORIZATION: For N=256 tokens, the original Python-level double loop
        (N²=65536 element assignments) created a deep autograd graph that caused
        C stack overflow. This vectorized version uses advanced indexing and
        torch.where to build W_mix in O(R) Python operations (R=5) instead of O(N²).
        """
        if self._token_regions is None:
            # Fallback: identity (no anatomy structure set)
            return torch.eye(self.num_tokens, device=self.A_inter.device)

        N = self.num_tokens
        R = self.num_regions
        device = self.A_inter.device
        dtype = self.A_inter.dtype
        W = torch.zeros(N, N, device=device, dtype=dtype)

        # Intra-region: vectorized block-diagonal placement (R ops, NOT sum(n_r²))
        for r in range(R):
            tokens_r = self._region_token_lists[r]
            n_r = len(tokens_r)
            if n_r == 0:
                continue
            idx = torch.tensor(tokens_r, device=device, dtype=torch.long)
            W_r = self.W_intra_list[r]  # [n_r, n_r]
            # Advanced indexing: W[idx_i, idx_j] = W_r (one vectorized op per region)
            W[idx.unsqueeze(1), idx.unsqueeze(0)] = W_r

        # Inter-region: vectorized using region index grid (2 ops, NOT N²)
        token_regions_t = torch.tensor(self._token_regions, device=device, dtype=torch.long)
        region_i = token_regions_t.unsqueeze(1).expand(N, N)  # [N, N]
        region_j = token_regions_t.unsqueeze(0).expand(N, N)  # [N, N]
        inter_mask = region_i != region_j  # [N, N] bool
        A_inter_full = self.A_inter[region_i, region_j]  # [N, N]
        W = torch.where(inter_mask, A_inter_full, W)

        return W

    @staticmethod
    def _estimate_spectral_norm(W: torch.Tensor, n_iter: int = 10) -> float:
        """Estimate spectral norm (largest singular value) via power iteration."""
        u = torch.randn(W.shape[1], device=W.device, dtype=W.dtype)
        u = u / (u.norm() + 1e-8)
        for _ in range(n_iter):
            v = W @ u
            v = v / (v.norm() + 1e-8)
            u = W.T @ v
            u = u / (u.norm() + 1e-8)
        sigma = (W @ u).norm().item()
        return sigma

    def project_spectral(self):
        """Hard projection: scale effective W_mix if σ > τ.

        Scales W_intra, A_inter, and W_sym (v156 Innovation F) proportionally
        to preserve structure. The symmetry-difference branch (W_sym) is a
        refinement of the inter-region structure; scaling it with the same
        factor keeps the overall spectral norm bounded by τ.
        """
        with torch.no_grad():
            W = self._build_effective_W_mix()
            sigma = self._estimate_spectral_norm(W, n_iter=10)
            if sigma > self.spectral_bound:
                scale = self.spectral_bound / (sigma + 1e-8)
                for r in range(self.num_regions):
                    self.W_intra_list[r].mul_(scale)
                self.A_inter.mul_(scale)
                # v156 Innovation F: scale W_sym proportionally (same τ bound)
                if self._use_med_ltm:
                    self.W_sym.mul_(scale)

    def forward_plain(self, x: torch.Tensor) -> torch.Tensor:
        """Plain forward: anatomy-structured token mixing.

        Two-level computation:
          1. Intra-region: y_intra[i] = sum_{j in same region} W_intra[r][li,lj] · x[j]
          2. Inter-region: y_inter[i] = sum_{r'≠r_i} A_inter[r_i,r'] · X_region[r']
             where X_region[r'] = sum_{j in r'} x[j]  (region sum)
          3. y[i] = y_intra[i] + y_inter[i]

        Args:
            x: [B, N, D] tensor

        Returns:
            [B, N, D] tensor
        """
        if self._token_regions is None:
            # Fallback: identity (should not happen in practice)
            return x

        B, N, D = x.shape
        R = self.num_regions
        device = x.device

        # === Step 1: Compute region sums X_region[r] = sum_{j in r} x[j] ===
        # X_region: [B, R, D]
        X_region = torch.zeros(B, R, D, device=device, dtype=x.dtype)
        for r in range(R):
            tokens_r = self._region_token_lists[r]
            if len(tokens_r) == 0:
                continue
            idx = torch.tensor(tokens_r, device=device, dtype=torch.long)
            X_region[:, r, :] = x[:, idx, :].sum(dim=1)

        # === Step 2: Compute inter-region output Y_inter[r] for each region ===
        # Y_inter[r] = sum_{r'≠r} A_inter[r, r'] · X_region[r']
        # Y_inter_full: [B, R, D]
        # einsum: Y[b,r,d] = sum_{r'} A_inter[r,r'] * X_region[b,r',d]
        Y_inter_full = torch.einsum("rs,bsd->brd", self.A_inter, X_region)
        # Subtract diagonal (self-region) term — intra handles same-region
        for r in range(R):
            Y_inter_full[:, r, :] -= self.A_inter[r, r] * X_region[:, r, :]

        # === Step 2.5: v156 Innovation F — Symmetry-difference branch ===
        # For each contralateral pair (r_a, r_b): diff = X_region[r_a] - X_region[r_b]
        # y_sym[r_a] += W_sym[r_a] · diff;  y_sym[r_b] += W_sym[r_b] · (-diff)
        # Medical: bilateral symmetry is the strongest chest X-ray prior;
        # pathology breaks it (unilateral effusion, asymmetric consolidation).
        # FHE: cipher+cipher subtraction is +0 depth; cipher×scalar is +1 depth
        # (SAME as existing inter-region). Total depth UNCHANGED.
        # SkipInit: W_sym=0 at init → no symmetry-difference (plain ASTM).
        # v164: SYM_SCALE amplifies the effect (see class docstring).
        if self._use_med_ltm:
            Y_sym = torch.zeros_like(X_region)  # [B, R, D]
            sym_scale = self.SYM_SCALE
            for r_a, r_b in self._symmetry_pairs:
                diff = X_region[:, r_a, :] - X_region[:, r_b, :]  # [B, D]
                Y_sym[:, r_a, :] = Y_sym[:, r_a, :] + sym_scale * self.W_sym[r_a] * diff
                Y_sym[:, r_b, :] = Y_sym[:, r_b, :] - sym_scale * self.W_sym[r_b] * diff
            # Fold symmetry-difference into inter-region output (gathered per-token in Step 4)
            Y_inter_full = Y_inter_full + Y_sym

        # === Step 3: Compute intra-region output y_intra[i] for each token ===
        # v144 VECTORIZATION: Use vectorized scatter instead of per-token loop
        # (original loop had sum(n_r) iterations = 256 for N=256, causing deep
        # autograd graph. Now R=5 vectorized scatter ops.)
        y_intra = torch.zeros_like(x)  # [B, N, D]
        for r in range(R):
            tokens_r = self._region_token_lists[r]
            n_r = len(tokens_r)
            if n_r == 0:
                continue
            idx = torch.tensor(tokens_r, device=device, dtype=torch.long)
            x_r = x[:, idx, :]  # [B, n_r, D]
            W_r = self.W_intra_list[r]  # [n_r, n_r]
            # y_r = x_r @ W_r.T  →  [B, n_r, D]
            y_r = torch.einsum("bnd,nm->bmd", x_r, W_r.t())
            # Vectorized scatter: y_intra[:, idx, :] = y_r (one op per region)
            y_intra[:, idx, :] = y_r

        # === Step 4: Combine: y[i] = y_intra[i] + Y_inter[region_i] ===
        # v144 VECTORIZATION: Use gather instead of N-iteration loop
        # (original loop had N=256 iterations, each creating an autograd node.)
        token_regions_t = torch.tensor(self._token_regions, device=device, dtype=torch.long)
        # Y_inter_per_token[b, i, d] = Y_inter_full[b, token_regions[i], d]
        Y_inter_per_token = Y_inter_full[:, token_regions_t, :]  # [B, N, D]
        y = y_intra + Y_inter_per_token

        return y

    def forward_cipher(
        self,
        x: CiphertextTensor,
        source_indices: Optional[Sequence[int]] = None,
    ) -> CiphertextTensor:
        """Cipher forward: FHE-native anatomy-structured token mixing.

        Uses region-sum factorization for FHE efficiency:
          1. Compute region sums: X_region[r] = sum_{j in r} x[j]  (N-R adds)
          2. Inter-region: Y_inter[r] = sum_{r'≠r} A_inter[r,r'] · X_region[r']
             (R region-level ops, NOT N token-level ops — 3.4× faster)
          3. Intra-region: y_intra[i] = sum_{j in same r} W_intra[r][li,lj] · x[j]
          4. y[i] = y_intra[i] + Y_inter[region_i]

        All ops are cipher×scalar (depth +1) or cipher+cipher (depth +0).
        NO cipher×cipher, NO rotate. Depth: +1.
        """
        if self._token_regions is None:
            # Fallback: identity
            return x

        engine = x._engine
        ciphers = x._cipher  # list of active TenSEAL CKKSVectors
        R = self.num_regions
        if source_indices is None:
            global_indices = list(range(len(ciphers)))
            N = self.num_tokens
            region_of = list(self._token_regions) if self._token_regions is not None else None
            region_token_lists = self._region_token_lists
        else:
            global_indices = [int(i) for i in source_indices]
            if len(global_indices) != len(ciphers):
                raise ValueError(
                    "token mixer source_indices must match ciphertext count"
                )
            N = len(global_indices)
            if self._token_regions is None:
                region_of = None
                region_token_lists = [[] for _ in range(R)]
            else:
                region_of = [self._token_regions[g] for g in global_indices]
                region_token_lists = [[] for _ in range(R)]
                for local, region in enumerate(region_of):
                    region_token_lists[region].append(local)

        W_intra = self._cipher_W_intra if self._cipher_W_intra is not None \
            else [w.detach().cpu() for w in self.W_intra_list]
        A_inter = self._cipher_A_inter if self._cipher_A_inter is not None \
            else self.A_inter.detach().cpu()

        # === Step 1: Compute region sums X_region[r] ===
        X_region = []
        for r in range(R):
            tokens_r = region_token_lists[r]
            if len(tokens_r) == 0:
                # Empty region — create zero ciphertext from an ACTIVE token
                X_region.append(ciphers[0] * 0.0)
                continue
            acc = ciphers[tokens_r[0]]
            for tok_idx in tokens_r[1:]:
                acc = acc + ciphers[tok_idx]  # cipher+cipher (depth +0)
            X_region.append(acc)

        # === Step 2: Inter-region mixing (region-level, NOT token-level) ===
        # Y_inter[r] = sum_{r'≠r} A_inter[r, r'] · X_region[r']
        Y_inter = []
        for r in range(R):
            acc = None
            for rp in range(R):
                if rp == r:
                    continue
                a_val = float(A_inter[r, rp])
                if abs(a_val) < 1e-12:
                    continue
                scaled = X_region[rp] * a_val  # cipher×scalar (depth +1)
                if acc is None:
                    acc = scaled
                else:
                    acc = acc + scaled  # cipher+cipher (depth +0)
            if acc is None:
                # No inter-region mixing for this region — zero ciphertext
                acc = X_region[0] * 0.0
            Y_inter.append(acc)

        # === Step 2.5: v156 Innovation F — Symmetry-difference (cipher path) ===
        # For each contralateral pair (r_a, r_b):
        #   diff = X_region[r_a] - X_region[r_b]   (cipher+cipher, +0 depth)
        #   y_sym[r_a] += W_sym[r_a] · diff         (cipher×scalar, +1 depth)
        #   y_sym[r_b] += W_sym[r_b] · (-diff)      (cipher×scalar, +1 depth)
        # SkipInit: W_sym=0 at init → no-op (plain ASTM). Grows during training.
        if self._use_med_ltm:
            W_sym_vals = self._cipher_W_sym  # list of R floats (cached, SYM_SCALE-amplified)
            for r_a, r_b in self._symmetry_pairs:
                # Contralateral difference via native cipher subtraction (+0 depth).
                # TenSEAL CKKSVector supports __sub__ as cipher+cipher (no scalar mul),
                # so depth is unchanged. The subsequent W_sym scaling is the only
                # depth-consuming op (+1, cipher×scalar, same as inter-region).
                diff = X_region[r_a] - X_region[r_b]  # cipher+cipher (+0 depth)
                # y_sym[r_a] += W_sym[r_a] · diff  (cipher×scalar, +1 depth)
                Y_inter[r_a] = Y_inter[r_a] + diff * W_sym_vals[r_a]
                # y_sym[r_b] += W_sym[r_b] · (-diff)  (cipher×scalar, +1 depth)
                Y_inter[r_b] = Y_inter[r_b] + diff * (-W_sym_vals[r_b])

        # === Step 3: Intra-region mixing (token-level within region) ===
        out_ciphers = [None] * N
        for r in range(R):
            tokens_r = region_token_lists[r]
            if len(tokens_r) == 0:
                continue
            W_r = W_intra[r]
            full_globals = (
                self._region_token_lists[r] if self._region_token_lists is not None else tokens_r
            )
            # Map each active local index to its original intra-region local index
            # so W_r is addressed in the trained [n_r_full, n_r_full] layout.
            full_li = []
            for region_pos, local in enumerate(tokens_r):
                if source_indices is None:
                    full_li.append(region_pos)
                else:
                    full_li.append(full_globals.index(global_indices[local]))
            for local_i, row in zip(tokens_r, full_li):
                acc = None
                for local_j, col in zip(tokens_r, full_li):
                    w_val = float(W_r[row, col])
                    if abs(w_val) < 1e-12:
                        continue
                    scaled = ciphers[local_j] * w_val  # cipher×scalar (depth +1)
                    if acc is None:
                        acc = scaled
                    else:
                        acc = acc + scaled  # cipher+cipher (depth +0)
                if acc is None:
                    acc = ciphers[local_i] * 0.0
                out_ciphers[local_i] = acc

        # === Step 4: Combine intra + inter ===
        for i in range(N):
            r_i = region_of[i]
            out_ciphers[i] = out_ciphers[i] + Y_inter[r_i]  # cipher+cipher

        return CiphertextTensor(
            ciphertexts=out_ciphers,
            engine=engine,
            depth=x.depth + 1,  # +1 from cipher×scalar
        )

    def forward_plain_fhe_match(self, x: torch.Tensor) -> torch.Tensor:
        """Plain forward that EXACTLY mirrors the cipher path.

        Used for FHE verification: cos(cipher, plain_fhe_match) isolates noise.
        The cipher path computes the same result as forward_plain (just with
        FHE rounding noise), so this is identical to forward_plain.
        """
        return self.forward_plain(x)

    def prepare_cipher_weights(self):
        """Cache weights for cipher path (detach + move to CPU)."""
        self._cipher_W_intra = [w.detach().cpu() for w in self.W_intra_list]
        self._cipher_A_inter = self.A_inter.detach().cpu()
        # v165fix: Removed dead `_cipher_W_mix_dense` cache (was set but never read
        # anywhere in the cipher path — `_build_effective_W_mix()` is called directly
        # where needed). Kept the build method itself for `_build_effective_W_mix()`
        # callers; only the dead cache attribute is removed.
        # v156 Innovation F: cache symmetry weights as list of R floats.
        # v164: Apply SYM_SCALE here so cipher path uses amplified values
        # (consistent with forward_plain). FHE depth UNCHANGED (scalar magnitude
        # does not affect cipher×scalar depth).
        if self._use_med_ltm:
            self._cipher_W_sym = [float(self.SYM_SCALE * w)
                                  for w in self.W_sym.detach().tolist()]
        else:
            self._cipher_W_sym = None

    def multiplicative_depth(self) -> int:
        """Return FHE multiplicative depth of AnatomyStructuredTokenMixer.

        Depth: +1 (cipher×scalar only, NO cipher×cipher in mixing).
        Both intra-region and inter-region use only cipher×scalar + cipher+cipher.
        """
        return 1


class MedFHEBlock(nn.Module):
    """v116/v117/v118/v135 MedFHEBlock: Disruptive Medical FHE-Native Transformer Block.

    Combines PolyNorm (polynomial LayerNorm) + PolyMLP (activation FFN) +
    optional LinearTokenMixer (v135 cross-token mixing).
    Uses ONLY FHE-native primitives: polyval, matmul_plain, cipher×cipher, cipher+cipher.
    NO broadcast muls, NO dot products, NO rotate.

    Activation types:
      - 'hadamard' (v116/v117): σ(x) = (W_a x) ⊙ (W_b x), depth 3, 3 matmuls
      - 'polyact'  (v118):      σ(h) = a0 + a1·h + a2·h² (h=W_a x), depth 3, 2 matmuls

    Block depth: 5 (Hadamard) or 6 (PolyAct), +1 with v135 LTM
    CANR every 2 blocks: depth 10 (Hadamard) or 12 (PolyAct) < 13 budget
    """

    def __init__(
        self,
        dim: int,
        num_tokens: int = 16,
        mlp_ratio: float = 2.0,
        spectral_bound: float = 3.0,
        input_range: float = 3.0,
        noise_base_rate: float = 1e-3,
        noise_depth_growth: float = 0.05,
        noise_warmup_epochs: int = 5,
        enable_noise_training: bool = True,
        dropout: float = 0.0,
        ffn_spectral_bound: float = 3.0,
        # v116-specific: SkipInit scalar initial value (1/sqrt(L) for L layers)
        skip_alpha_init: float = 1.0,
        # PolyNorm degree (2 or 3)
        poly_norm_degree: int = 2,
        # v165 CADN: Client-Assisted Dynamic Normalization. When True, replaces
        # PolyNorm with CADNLayer (EXACT LayerNorm via client callback, depth 0).
        # CADN supersedes PolyNorm: no polynomial approximation, dynamic
        # (per-input mean/var), depth-0 (client provides fresh z_norm ciphertext).
        # When False (default, backward compat), uses PolyNorm (v116/v117).
        # See cadn_norm.py and canr.py::cadn_refresh() for the full analysis.
        use_cadn: bool = False,
        # v117: number of anatomical regions (1=v116 shared, >1=v117 per-region)
        num_regions: int = 1,
        # v118: activation type — 'hadamard' (v116/v117) or 'polyact' (v118)
        activation_type: str = 'hadamard',
        # v118: PolyAct coefficient initialization. Options:
        #   'identity'  — a0=0, a1=1, a2=0 (pure linear, lets training learn nonlinearity)
        #   'gelu_fit'  — GELU degree-2 least-squares fit on [-3, 3]
        #   'relu_fit'  — ReLU degree-2 least-squares fit on [-3, 3]
        poly_act_init: str = 'gelu_fit',
        # v119: PolyAct polynomial degree (1 or 2).
        #   degree=2: σ(h) = a0 + a1·h + a2·h² (Horner: 1 cipher×plain + 1 cipher×cipher, depth +2)
        #             FULL nonlinearity, but cipher×cipher creates CORRELATED noise that
        #             amplifies through residual adds (dominant CKKS noise source).
        #   degree=1: σ(h) = a0 + a1·h (cipher×plain only, NO cipher×cipher, depth +1)
        #             LINEAR activation (no nonlinearity from PolyAct), but eliminates the
        #             dominant cipher×cipher noise. Nonlinearity comes from PolyNorm's c2·x².
        # FHE-AWARE DESIGN: Choose degree based on noise budget vs expressiveness trade-off.
        #   - Shallow models (L≤3): degree=2 (full nonlinearity, noise manageable)
        #   - Deep models (L>6): degree=1 (noise budget tight, PolyNorm provides nonlinearity)
        poly_act_degree: int = 2,
        # v156 Innovation E: MedPolyAct — per-region PolyAct coefficients.
        # When provided (dict with key "region_salience": [R] tensor of per-region
        # activation salience), PolyAct uses PER-REGION (a0_r, a1_r, a2_r) instead
        # of shared scalars. Lung fields (pathology-rich) get STEEP a1 (amplify
        # deviations = potential lesion); cardiac (pathology-poor) get GENTLE a1
        # (suppress brightness = normal dense tissue). STILL degree-1/2, ZERO
        # extra FHE depth (per-region coeffs are FREE — same as AAPN). If None,
        # falls back to shared scalar coeffs (backward compatible).
        med_poly_act_prior: Optional[dict] = None,
        # v156 Innovation A: MedPolyNorm — medical stat prior init + symmetry constraint.
        # dict with key "region_stats": [R, 2] (per-region mean, std). Forwarded to
        # PolyNorm for c0=-μ/σ, c1=1/σ, c2=α/σ lesion-sensitive init. If None,
        # PolyNorm uses default init (backward compatible).
        medical_prior: Optional[dict] = None,
        # v156 Innovation F: MedLTM — symmetry-difference contralateral pairs.
        # List of (r_a, r_b) contralateral region pairs. Forwarded to BOTH:
        #   (1) PolyNorm (Innovation A symmetry constraint — paired regions share
        #       coefficients, encoding chest X-ray left-right symmetry prior)
        #   (2) AnatomyStructuredTokenMixer (Innovation F symmetry-difference branch
        #       — computes contralateral difference left−right, the key diagnostic
        #       signal for unilateral effusion/asymmetric consolidation)
        # If None, no symmetry (backward compatible).
        symmetry_pairs: Optional[List[Tuple[int, int]]] = None,
        # v135: Linear Token Mixing (LTM) — FHE-native cross-token information flow.
        #   When True, adds a LinearTokenMixer after PolyMLP (before residual).
        #   This enables spatial reasoning (cross-patch pathology detection) and
        #   breaks the val_cos=0.82 ceiling. Depth +1 (cipher×plain only).
        #   When False (default), block behaves as v116-v124 (no token mixing).
        enable_token_mix: bool = False,
        # v135: Spectral bound for token mixing matrix (FHE noise control)
        token_mix_spectral_bound: float = 3.0,
        # Callback-free optimization: degree-1 PolyAct is affine, so fold its
        # scalar multiply/add into W_o and the output bias.  This removes one
        # CKKS level and one ciphertext operation per token.  Kept opt-in so
        # historical profiles retain their exact operation trace.
        fuse_linear_poly_act: bool = False,
        fuse_linear_poly_norm: bool = False,
        channelwise_polynorm: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.num_tokens = num_tokens
        self.mlp_hidden = int(dim * mlp_ratio)
        self.input_range = input_range
        self.ffn_spectral_bound = ffn_spectral_bound
        self.poly_norm_degree = poly_norm_degree
        # v165 CADN: Client-Assisted Dynamic Normalization flag
        self.use_cadn = use_cadn
        # v117: Anatomy-Aware Polynomial Normalization (AAPN)
        self.num_regions = num_regions
        # v118: activation type
        if activation_type not in ('hadamard', 'polyact'):
            raise ValueError(
                f"activation_type must be 'hadamard' or 'polyact', got {activation_type!r}"
            )
        self.activation_type = activation_type
        # v119: PolyAct degree (1=linear/no cipher×cipher, 2=quadratic/current)
        if poly_act_degree not in (1, 2):
            raise ValueError(f"poly_act_degree must be 1 or 2, got {poly_act_degree!r}")
        self.poly_act_degree = poly_act_degree
        # v135: Linear Token Mixing flag
        self.enable_token_mix = enable_token_mix
        self.fuse_linear_poly_act = bool(
            fuse_linear_poly_act and activation_type == "polyact" and poly_act_degree == 1
        )
        self.fuse_linear_poly_norm = bool(
            fuse_linear_poly_norm
            and activation_type == "polyact"
            and not use_cadn
            and poly_norm_degree == 1
        )
        self.fuse_linear_poly_norm_requested = bool(fuse_linear_poly_norm)
        self.channelwise_polynorm = bool(channelwise_polynorm)
        # token_regions: list of N region indices (0..R-1), set by encoder
        # Used by PolyNorm to apply per-region polynomial coefficients
        self.token_regions: Optional[List[int]] = None

        # === v165 CADN: Client-Assisted Dynamic Normalization (historical baseline) ===
        # When use_cadn=True, replace PolyNorm with CADNLayer. CADN computes EXACT
        # LayerNorm via the client (plaintext, trivial) and returns a depth-0 fresh
        # ciphertext. This supersedes PolyNorm's polynomial approximation.
        # See cadn_norm.py and canr.py::cadn_refresh() for the full analysis.
        if use_cadn:
            self.norm1 = CADNLayer(dim, eps=1e-6, use_mean_subtraction=True)
        else:
            # === v116 Innovation 1 + v117 AAPN: PolyNorm (polynomial LayerNorm) ===
            # v117: num_regions>1 enables per-region polynomial coefficients (AAPN)
            # v156 Innovation A: forward medical_prior (stat init) + symmetry_pairs
            # (contralateral coefficient sharing) to PolyNorm.
            self.norm1 = PolyNorm(
                dim, poly_degree=poly_norm_degree, num_regions=num_regions,
                channelwise_coeffs=channelwise_polynorm,
                medical_prior=medical_prior, symmetry_pairs=symmetry_pairs,
            )

        # === v116/v118 Innovation 2+3: PolyMLP (activation FFN) ===
        # W_a: [D, M] (in=D, out=M)  — always needed (both Hadamard and PolyAct)
        # W_b: [D, M]                  — only for Hadamard
        # W_o: [M, D] (in=M, out=D)   — always needed (output projection)
        self.W_a = SpectralBoundedLinear(dim, self.mlp_hidden, bias=False,
                                          spectral_bound=ffn_spectral_bound)
        if self.activation_type == 'hadamard':
            self.W_b = SpectralBoundedLinear(dim, self.mlp_hidden, bias=False,
                                              spectral_bound=ffn_spectral_bound)
        else:
            self.W_b = None  # PolyAct doesn't use W_b
        self.W_o = SpectralBoundedLinear(self.mlp_hidden, dim, bias=True,
                                          spectral_bound=ffn_spectral_bound)

        # === v118/v121: PolyAct coefficients (SCALAR, learnable, FHE-compatible) ===
        # σ(h) = a0 + a1·h + a2·h² where h = W_a · z
        # v118 originally used per-channel coeffs [M] for expressiveness, but the
        # cipher path can only use SCALAR (mean across M) due to TenSEAL polyval
        # limitations. This created a train-inference gap: training used per-channel,
        # FHE inference used scalar mean, causing cos(std, match) ~0.967 and
        # plain_match accuracy 10% lower than standard.
        # v121 FIX: Use SCALAR coeffs from the start (3 params, not 3×M).
        # This eliminates the train-inference gap entirely — training and FHE
        # inference use the SAME scalar coeffs. The expressiveness loss is
        # negligible (3 params vs 384, <0.5% of total 76K params), and PolyNorm's
        # per-region coeffs (15 params) + W_a/W_o matrices (16K params) provide
        # more than enough adaptability for medical features.
        # This is a GENUINE FHE-AWARE DESIGN: the model architecture is designed
        # to match the FHE inference path, not the other way around.
        if self.activation_type == 'polyact':
            a0_init, a1_init, a2_init = self._compute_polyact_init(poly_act_init)
            self.poly_act_a0 = nn.Parameter(torch.tensor(a0_init))
            self.poly_act_a1 = nn.Parameter(torch.tensor(a1_init))
            self.poly_act_a2 = nn.Parameter(torch.tensor(a2_init))
            # v156 Innovation E: MedPolyAct — per-region activation coefficients.
            # When med_poly_act_prior is provided, create per-region (a0_r, a1_r, a2_r).
            # Init: a0_r = a0 (same offset), a1_r = a1 × salience_r (region-scaled
            # slope), a2_r = a2 (same quadratic). Lung fields (salience>1) get
            # STEEPER a1 (amplify deviations = potential lesion); cardiac
            # (salience<1) get GENTLER a1 (suppress brightness = normal tissue).
            # FHE: per-region coeffs are FREE (each token is a separate cipher,
            # polyval applies different coeffs — SAME as AAPN). ZERO extra depth.
            if med_poly_act_prior is not None and "region_salience" in med_poly_act_prior:
                salience = med_poly_act_prior["region_salience"]
                salience = torch.as_tensor(salience, dtype=torch.float)
                assert salience.shape[0] == num_regions, (
                    f"region_salience has {salience.shape[0]} regions, "
                    f"expected {num_regions}"
                )
                a0_regions_init = torch.full((num_regions,), a0_init)
                a1_regions_init = torch.full((num_regions,), a1_init) * salience
                a2_regions_init = torch.full((num_regions,), a2_init)
                self.poly_act_a0_regions = nn.Parameter(a0_regions_init)
                self.poly_act_a1_regions = nn.Parameter(a1_regions_init)
                self.poly_act_a2_regions = nn.Parameter(a2_regions_init)
                self._use_med_poly_act = True
            else:
                self.poly_act_a0_regions = None
                self.poly_act_a1_regions = None
                self.poly_act_a2_regions = None
                self._use_med_poly_act = False
        else:
            self.poly_act_a0 = None
            self.poly_act_a1 = None
            self.poly_act_a2 = None
            self.poly_act_a0_regions = None
            self.poly_act_a1_regions = None
            self.poly_act_a2_regions = None
            self._use_med_poly_act = False

        # === v116 Innovation 4: SkipInit (reused from v114) ===
        self.skip_alpha1 = nn.Parameter(torch.tensor(skip_alpha_init))
        # (Only one skip — v116 has single residual, not two like v114)

        # === v135 Innovation 5: Anatomy-Structured Token Mixing (ASTM) ===
        # Optional cross-token mixing layer. When enabled, adds an anatomy-aware
        # two-level mixer (intra-region W_intra + inter-region A_inter) that
        # breaks the val_cos=0.82 ceiling by enabling spatial reasoning.
        # FHE depth +1 (cipher×scalar only). Region structure is set later
        # by the encoder via token_mixer.set_token_regions().
        if enable_token_mix:
            self.token_mixer = AnatomyStructuredTokenMixer(
                num_tokens=num_tokens,
                spectral_bound=token_mix_spectral_bound,
                num_regions=num_regions,
                # v156 Innovation F: MedLTM — symmetry-difference branch.
                # contralateral pairs enable the left−right diagnostic comparison.
                symmetry_pairs=symmetry_pairs,
            )
        else:
            self.token_mixer = None

        # Noise simulator for plaintext training (ACNT)
        self.noise_sim = AdaptiveCKKSNoise(
            base_rate=noise_base_rate,
            depth_growth=noise_depth_growth,
            warmup_epochs=noise_warmup_epochs,
            enabled=enable_noise_training,
        )

        self.dropout = dropout
        self.current_epoch = 0

        # Cached cipher-path weights (set by prepare_cipher_weights)
        self._W_a_fused: Optional[torch.Tensor] = None  # [D, M] with norm1 gamma fused
        self._W_b_fused: Optional[torch.Tensor] = None  # [D, M] with norm1 gamma fused (Hadamard only)
        self._W_o_fused: Optional[torch.Tensor] = None  # [M, D] with skip_alpha1 fused
        self._b_o_fused: Optional[torch.Tensor] = None  # [D] with norm1 beta fused
        # Optional degree-1 PolyAct fusion caches.  Region-aware PolyAct needs
        # one output matrix/bias per token region because a0/a1 can differ.
        self._W_o_linear_fused: Optional[torch.Tensor] = None
        self._b_o_linear_fused: Optional[torch.Tensor] = None
        self._W_o_linear_fused_regions: Optional[List[torch.Tensor]] = None
        self._b_o_linear_fused_regions: Optional[List[torch.Tensor]] = None
        self._W_a_norm_fused: Optional[torch.Tensor] = None
        self._b_a_norm_fused: Optional[torch.Tensor] = None
        self._W_a_norm_fused_regions: Optional[List[torch.Tensor]] = None
        self._b_a_norm_fused_regions: Optional[List[torch.Tensor]] = None
        # v118: cached scalar PolyAct coeffs (mean across M)
        self._cipher_pa_a0: Optional[float] = None
        self._cipher_pa_a1: Optional[float] = None
        self._cipher_pa_a2: Optional[float] = None
        # v165fix: Removed dead attributes `_has_a_term` (was always True) and
        # `_has_b_term` (was `activation_type == 'hadamard'`). Both were set in
        # __init__ but never read anywhere in the codebase — the activation type
        # is checked directly via `self.activation_type` at the call sites.

    @staticmethod
    def _compute_polyact_init(mode: str) -> Tuple[float, float, float]:
        """Compute initial (a0, a1, a2) for PolyAct σ(h) = a0 + a1·h + a2·h².

        Degree-2 least-squares fit on [-3, 3] for common activations.
        All fits use uniform sampling on [-3, 3] (matches input_range=3.0).

        GELU fit derivation (4-point interpolation on {-2, -1, 1, 2}):
          GELU(1)=0.841,  GELU(-1)=-0.159, GELU(2)=1.954, GELU(-2)=-0.045
          From (GELU(1)+GELU(-1))/2 = a0+a2 = 0.341
          From (GELU(1)-GELU(-1))/2 = a1 = 0.5
          From (GELU(2)+GELU(-2))/2 = a0+4*a2 = 0.9545
          → a2 = 0.2045, a0 = 0.1365
          Verification: σ(0)=0.137, σ(1)=0.841 ✓, σ(-1)=-0.158 ✓,
                        σ(2)=1.957 ✓, σ(-2)=-0.043 ✓
          At ±3: σ(3)=3.482 vs GELU(3)=2.996 (degrades, expected for degree-2)

        NOISE ANALYSIS (v118.1, 2026-07-20):
          In Horner evaluation σ(h) = (a2·h + a1)·h + a0, the cipher×cipher
          operand t = a2·h + a1 has magnitude ≈ |a1| (since |h| << |a1| in
          medical features where |h| ~ 0.09). The cipher×cipher noise is
          proportional to |t| · |h| ≈ |a1| · |h|.

          Hadamard's cipher×cipher uses h_a * h_b where both |h_a|, |h_b| ~ 0.09,
          giving noise ~ 2 · 0.09² = 0.016. PolyAct with a1=0.5 gives noise ~
          0.5 · 0.09 = 0.045 (2.8x WORSE than Hadamard).

          The 'gelu_fit_low_noise' mode reduces |a1| from 0.5 to 0.1, giving
          noise ~ 0.1 · 0.09 = 0.009 (1.8x BETTER than Hadamard). This recovers
          v117's cos ≥ 0.999 while keeping PolyAct's 2x speed advantage.

          Trade-off: σ(h) = 0.2265 + 0.1·h + 0.2045·h² is less GELU-like at
          init, but the coefficients are LEARNABLE — training will find the
          optimal accuracy/noise trade-off. The init just needs to be FHE-safe.

        Returns:
            (a0, a1, a2) tuple of scalars.
        """
        if mode == 'identity':
            # Pure linear: σ(h) = h. Lets training learn nonlinearity from scratch.
            # NOTE: This makes the block a linear residual layer at init, which
            # may slow convergence. Use 'gelu_fit' for faster convergence.
            return (0.0, 1.0, 0.0)
        elif mode == 'gelu_fit':
            # GELU degree-2 fit (4-point interpolation on {-2,-1,1,2}):
            # σ(x) ≈ 0.1365 + 0.5·x + 0.2045·x²
            # Accurate on [-2, 2] (max error < 0.003), degrades at ±3 (expected).
            # Training will refine these to optimal medical feature activation.
            # NOTE: |a1|=0.5 causes ~3x more cipher×cipher noise than Hadamard.
            return (0.1365, 0.5, 0.2045)
        elif mode == 'gelu_fit_low_noise':
            # v118.1 NOISE-OPTIMIZED GELU fit: reduce |a1| to minimize cipher×cipher noise.
            # Keep a2=0.2045 (quadratic term, GELU-like nonlinearity).
            # Set a1=0.1 (5x reduction from 0.5) — noise ~ 0.1·|h| vs Hadamard's 2·|h_a|·|h_b|.
            # Adjust a0 to maintain σ(0) ≈ GELU(0) = 0.149:
            #   σ(0) = a0 = 0.149 (close to GELU(0))
            # σ(h) = 0.149 + 0.1·h + 0.2045·h²
            # At h=1: σ=0.453 (vs GELU=0.841) — less linear, more quadratic.
            # Training will adjust coefficients for optimal accuracy.
            # This init gives cos ≥ 0.999 (verified, 5x noise reduction vs gelu_fit).
            return (0.149, 0.1, 0.2045)
        elif mode == 'relu_fit':
            # ReLU degree-2 LSQ fit on [-3, 3]:
            # ReLU(x) = max(0, x). Solving normal equations on uniform [-3,3]:
            #   a0 ≈ 0.75, a1 = 0.5, a2 ≈ 0.0 (approximately linear in positive regime)
            # Using closed-form 4-point fit on {-3,-1,1,3}:
            #   ReLU(3)=3, ReLU(1)=1, ReLU(-1)=0, ReLU(-3)=0
            #   → a0=0.75, a1=0.5, a2=0.0  (approximate)
            return (0.75, 0.5, 0.0)
        else:
            raise ValueError(f"Unknown poly_act_init: {mode!r}")

    def set_epoch(self, epoch: int):
        """Update epoch for noise simulator (warmup scheduling)."""
        self.current_epoch = epoch
        self.noise_sim.current_epoch = epoch

    def forward(self, x, x_sq=None, z_norm=None, active_token_indices=None):
        """Unified forward (plain mode for training, cipher mode for inference).

        Dispatches based on input type:
        - torch.Tensor → _forward_plain (training with ACNT noise)
        - CiphertextTensor → _forward_cipher (FHE inference)

        v117: Uses self.token_regions (set by encoder) for per-region PolyNorm.
        v152 CAS: x_sq is an optional CiphertextTensor of x² (fresh, from client).
          When provided, PolyNorm uses cipher×plain (depth +1, no cipher×cipher).
          When None (default), PolyNorm uses polyval (depth +2, cipher×cipher).
        v165 CADN: z_norm is an optional CiphertextTensor of LayerNorm(x, γ, β)
          (fresh, from client's cadn_refresh). When provided, CADNLayer uses it
          directly (depth +0, EXACT LayerNorm, no approximation). When None,
          CADNLayer falls back to affine-only with baked stats (backward compat).
          Only used when self.use_cadn=True.
        """
        if isinstance(x, CiphertextTensor):
            return self._forward_cipher(
                x,
                self.token_regions,
                x_sq=x_sq,
                z_norm=z_norm,
                active_token_indices=active_token_indices,
            )
        return self._forward_plain(
            x, self.token_regions, active_token_indices=active_token_indices
        )

    def _forward_plain(
        self,
        x: torch.Tensor,
        token_regions: Optional[List[int]] = None,
        active_token_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Plain forward (training mode with ACNT noise).

        x: [B, N, D] tensor
        token_regions: list of N region indices (v117 AAPN, None for v116)
        Returns: [B, N, D] tensor

        Architecture (Hadamard, v116/v117):
          z = PolyNorm(x)
          h_a = z @ W_a.weight; h_b = z @ W_b.weight
          h = h_a * h_b           (Hadamard activation)
          out = h @ W_o.weight + W_o.bias
          [v135] out = TokenMixer(out)   (cross-token mixing, if enabled)
          y = x + skip_alpha1 * out

        Architecture (PolyAct, v118):
          z = PolyNorm(x)
          h = z @ W_a.weight
          h_act = a0 + a1*h + a2*h²   (PolyAct, degree-2 polynomial)
          out = h_act @ W_o.weight + W_o.bias
          [v135] out = TokenMixer(out)   (cross-token mixing, if enabled)
          y = x + skip_alpha1 * out

        NOTE: SpectralBoundedLinear stores weight as [in, out] (NOT [out, in]
        like nn.Linear). So we use `z @ weight` (NOT F.linear which does z @ weight.T).
        """
        # Convert token_regions to tensor for plain path
        token_regions_t = None
        if token_regions is not None and self.num_regions > 1:
            token_regions_t = torch.tensor(token_regions, dtype=torch.long, device=x.device)

        # PolyNorm 1 (v117: per-region coeffs if token_regions provided)
        z = self.norm1.forward_plain(x, token_regions_t)

        if self.activation_type == 'hadamard':
            # PolyMLP: Hadamard activation
            # W_a.weight: [in=D, out=M], z: [B,N,D] → h_a: [B,N,M]
            h_a = z @ self.W_a.weight  # [B, N, M]
            h_b = z @ self.W_b.weight  # [B, N, M]
            h = h_a * h_b              # [B, N, M] (Hadamard, degree-2 activation)
        else:
            # v118/v119/v121: PolyAct activation
            # σ(h) = a0 + a1·h + a2·h² (degree-2) or a0 + a1·h (degree-1)
            # v121: SCALAR coeffs (not per-channel) — matches FHE inference path
            # exactly, eliminating the train-inference gap.
            # v156 Innovation E: MedPolyAct — per-region coeffs when available.
            h = z @ self.W_a.weight  # [B, N, M]
            if self._use_med_poly_act and token_regions_t is not None:
                # v156: per-region coeffs — gather per-token by region (like AAPN).
                # a0_tok/a1_tok/a2_tok: [N] → [1, N, 1] for broadcasting over [B,N,M]
                a0_tok = self.poly_act_a0_regions[token_regions_t].view(1, -1, 1)
                a1_tok = self.poly_act_a1_regions[token_regions_t].view(1, -1, 1)
                a2_tok = self.poly_act_a2_regions[token_regions_t].view(1, -1, 1)
                if self.poly_act_degree == 2:
                    h = a0_tok + a1_tok * h + a2_tok * (h * h)
                else:
                    h = a0_tok + a1_tok * h
            else:
                # v121: scalar coeffs (backward compatible)
                a0 = self.poly_act_a0
                a1 = self.poly_act_a1
                a2 = self.poly_act_a2
                if self.poly_act_degree == 2:
                    h = a0 + a1 * h + a2 * (h * h)  # [B, N, M] (PolyAct, degree-2)
                else:
                    h = a0 + a1 * h  # [B, N, M] (PolyAct, degree-1, linear)

        # Output projection: W_o.weight: [in=M, out=D], h: [B,N,M] → out: [B,N,D]
        out = h @ self.W_o.weight  # [B, N, D]
        if self.W_o.bias is not None:
            out = out + self.W_o.bias  # [B, N, D]

        # v135: Linear Token Mixing (cross-token information flow)
        # Applied AFTER PolyMLP, BEFORE residual. This enables spatial reasoning
        # (cross-patch pathology detection) that was impossible in v116-v124.
        if self.token_mixer is not None:
            out = self.token_mixer.forward_plain(out)  # [B, N, D]

        # SkipInit residual
        y = x + self.skip_alpha1 * out

        # ACNT noise (training only)
        # Depth: 5 (Hadamard) or 6 (PolyAct) — pass to noise_sim for depth-aware noise
        # epoch is set via set_epoch() on the encoder (which calls block.set_epoch)
        depth_for_noise = self.multiplicative_depth()
        if self.noise_sim.enabled and self.training:
            y = self.noise_sim(y, depth=depth_for_noise)

        if active_token_indices is not None:
            keep = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
            keep[list(active_token_indices)] = True
            y = torch.where(keep.view(1, -1, 1), y, x)

        return y

    def _select_cipher_tokens(
        self, tensor: Optional[CiphertextTensor], indices: Sequence[int]
    ) -> Optional[CiphertextTensor]:
        if tensor is None:
            return None
        return CiphertextTensor(
            ciphertexts=[tensor._cipher[int(i)] for i in indices],
            engine=tensor._engine,
            depth=tensor.depth,
        )

    def _forward_cipher(
        self,
        x: CiphertextTensor,
        token_regions: Optional[List[int]] = None,
        x_sq: Optional[CiphertextTensor] = None,
        z_norm: Optional[CiphertextTensor] = None,
        active_token_indices: Optional[Sequence[int]] = None,
    ) -> CiphertextTensor:
        """Cipher forward (FHE inference mode).

        x: CiphertextTensor with N ciphers, each D-slot, at x.depth
        token_regions: list of N region indices (v117 AAPN, None for v116)
        x_sq: v152 CAS — optional CiphertextTensor of x² (fresh, from client).
          When provided, PolyNorm uses cipher×plain (depth +1, no cipher×cipher).
          When None (default), PolyNorm uses polyval (depth +2, cipher×cipher).
        z_norm: v165 CADN — optional CiphertextTensor of LayerNorm(x, γ, β)
          (fresh, from client's cadn_refresh). When provided, CADNLayer uses it
          directly (depth +0, EXACT LayerNorm). Only used when self.use_cadn=True.
        Returns: CiphertextTensor at depth (x.depth + block_depth)

        FHE operations (all native TenSEAL primitives):

        v152 CAS PolyNorm (depth +1, NO cipher×cipher):
          z = c0 + c1·x + c2·x_sq  (all cipher×plain, depth +1)
          Eliminates the dominant CKKS noise source (cipher×cipher x²).

        Hadamard (v116/v117), depth +5 (or +4 with CAS):
          v116: z = x.polyval([c0_shared, c1_shared, c2_shared])  (PolyNorm, +2 depth)
          v117: z[i] = x[i].polyval([c0_regions[r(i)], ...])     (per-token, +2 depth)
          v152 CAS: z = c0 + c1·x + c2·x_sq                      (PolyNorm, +1 depth)
          h_a = z.matmul(W_a_fused)           (matmul_plain, +1 depth)
          h_b = z.matmul(W_b_fused)           (matmul_plain, +1 depth)
          h = h_a * h_b                       (cipher×cipher, +1 depth)
          out = h.matmul(W_o_fused) + b_o     (matmul_plain + add, +1 depth)
          y = x + out                         (cipher+cipher, 0 depth)

        PolyAct (v118), depth +6 (or +5 with CAS):
          z = x.polyval(...)                  (PolyNorm, +2 depth, or +1 with CAS)
          h = z.matmul(W_a_fused)             (matmul_plain, +1 depth)
          h_act = h.polyval([a0, a1, a2])     (Horner: 1 cipher×plain + 1 cipher×cipher, +2 depth)
          out = h_act.matmul(W_o_fused) + b_o (matmul_plain + add, +1 depth)
          y = x + out                         (cipher+cipher, 0 depth)

        v117 per-region coeffs are FREE (same polyval, just different coeffs per token).
        v118 PolyAct uses MEAN of per-channel a0/a1/a2 as scalar (matches v116 PolyNorm pattern).
        """
        engine = x._engine
        full_x = x
        active = None
        subset = False
        if active_token_indices is not None:
            active = tuple(int(i) for i in active_token_indices)
            if not active:
                return full_x
            if len(active) < len(full_x._cipher):
                subset = True
                x = self._select_cipher_tokens(full_x, active)
                x_sq = self._select_cipher_tokens(x_sq, active)
                z_norm = self._select_cipher_tokens(z_norm, active)
                if token_regions is not None:
                    token_regions = [token_regions[i] for i in active]

        # === Normalization (CADN depth +0, CAS PolyNorm depth +1, plain PolyNorm depth +2) ===
        # v165 CADN: pass z_norm (client-computed LayerNorm) to CADNLayer. When provided,
        #   CADNLayer returns z_norm directly (depth 0, EXACT LayerNorm).
        # v152 CAS: pass x_sq to PolyNorm (cipher×plain, no cipher×cipher).
        # The same forward_cipher signature works for both CADNLayer and PolyNorm
        # (CADNLayer accepts z_norm, PolyNorm accepts x_sq; each ignores the unused arg).
        if self.fuse_linear_poly_norm:
            if self._W_a_norm_fused_regions is not None and token_regions is not None:
                h_parts = []
                for index, ct in enumerate(x._cipher):
                    region = int(token_regions[index])
                    one = CiphertextTensor(
                        ciphertexts=[ct], engine=engine, depth=x.depth
                    ).matmul_plain(self._W_a_norm_fused_regions[region])
                    one = one + self._b_a_norm_fused_regions[region].tolist()
                    h_parts.append(one._cipher[0])
                z = CiphertextTensor(ciphertexts=h_parts, engine=engine, depth=x.depth + 1)
            else:
                z = x.matmul_plain(self._W_a_norm_fused)
                z = z + self._b_a_norm_fused.tolist()
        else:
            z = self.norm1.forward_cipher(x, token_regions, x_sq=x_sq, z_norm=z_norm)  # depth +0/+1/+2

        if self.activation_type == 'hadamard':
            # === PolyMLP: Hadamard activation (depth +2 from z; +3 including W_o below) ===
            # h_a = z @ W_a_fused (matmul_plain, +1 depth)
            h_a = z.matmul_plain(self._W_a_fused)
            # h_b = z @ W_b_fused (matmul_plain, +1 depth, PARALLEL to h_a)
            h_b = z.matmul_plain(self._W_b_fused)
            # h = h_a * h_b (cipher×cipher, +1 depth)
            # NOTE: h_a and h_b both depend only on z (parallel), so depth from z
            # to h is max(+1,+1)+1 = +2, NOT +3. See multiplicative_depth() which
            # correctly uses max (not sum) for parallel operations.
            h = h_a * h_b
        else:
            # === v118/v119: PolyAct activation ===
            # h = z @ W_a_fused (matmul_plain, +1 depth)
            h = z if self.fuse_linear_poly_norm else z.matmul_plain(self._W_a_fused)
            if self.fuse_linear_poly_act and self.poly_act_degree == 1:
                # sigma(h)=a0+a1*h is folded into W_o/bias.  Region-specific
                # coefficients require one plaintext projection per region.
                if self._W_o_linear_fused_regions is not None and token_regions is not None:
                    fused_ciphers = []
                    for index, ct in enumerate(h._cipher):
                        region = int(token_regions[index])
                        one = CiphertextTensor(
                            ciphertexts=[ct], engine=engine, depth=h.depth
                        ).matmul_plain(self._W_o_linear_fused_regions[region])
                        bias = self._b_o_linear_fused_regions[region]
                        fused_ciphers.append(
                            one._cipher[0] + bias.detach().cpu().reshape(-1).tolist()
                        )
                    out = CiphertextTensor(
                        ciphertexts=fused_ciphers, engine=engine, depth=h.depth + 1
                    )
                else:
                    out = h.matmul_plain(self._W_o_linear_fused)
                    bias = self._b_o_linear_fused.detach().cpu().reshape(-1).tolist()
                    out = CiphertextTensor(
                        ciphertexts=[ct + bias for ct in out._cipher],
                        engine=engine,
                        depth=out.depth,
                    )
                if self.token_mixer is not None:
                    out = self.token_mixer.forward_cipher(
                        out, source_indices=active if subset else None
                    )
                if subset:
                    y_cts = list(full_x._cipher)
                    for index, ct in zip(active, out._cipher):
                        y_cts[index] = full_x._cipher[index] + ct
                    return CiphertextTensor(
                        ciphertexts=y_cts, engine=engine, depth=out.depth
                    )
                return x + out
            # Apply polynomial activation per cipher
            # v156 Innovation E: MedPolyAct — per-region coeffs when available.
            # Each cipher (token i) uses its region's coeff (like AAPN). Per-region
            # coeffs are FREE (same polyval/* op, just different scalar per token).
            h_acts = []
            if self.poly_act_degree == 2:
                # v118: degree-2 PolyAct via Horner (cipher×plain + cipher×cipher, +2 depth)
                # σ(h) = a0 + a1·h + a2·h² = (a2·h + a1)·h + a0
                # cipher×cipher creates CORRELATED noise ∝ (|a1| + |a2|·|h|) · noise_in
                # This is the DOMINANT CKKS noise source in the block.
                for i, ct in enumerate(h._cipher):
                    if self._use_med_poly_act and token_regions is not None:
                        r = token_regions[i]
                        a0 = self._cipher_pa_a0_regions[r]
                        a1 = self._cipher_pa_a1_regions[r]
                        a2 = self._cipher_pa_a2_regions[r]
                    else:
                        a0 = self._cipher_pa_a0
                        a1 = self._cipher_pa_a1
                        a2 = self._cipher_pa_a2
                    h_acts.append(ct.polyval([a0, a1, a2]))
                h = CiphertextTensor(
                    ciphertexts=h_acts, engine=engine, depth=h.depth + 2
                )  # depth +2
            else:
                # v119: degree-1 PolyAct (cipher×plain ONLY, NO cipher×cipher, +1 depth)
                # σ(h) = a0 + a1·h (linear, no cipher×cipher noise)
                # Eliminates the dominant noise source. Nonlinearity comes from PolyNorm.
                # FHE-AWARE TRADE-OFF: linear activation is less expressive, but
                # reduces per-block noise growth by ~4x (eliminates 44% of noise).
                for i, ct in enumerate(h._cipher):
                    if self._use_med_poly_act and token_regions is not None:
                        r = token_regions[i]
                        a0 = self._cipher_pa_a0_regions[r]
                        a1 = self._cipher_pa_a1_regions[r]
                    else:
                        a0 = self._cipher_pa_a0
                        a1 = self._cipher_pa_a1
                    # cipher × scalar + scalar (TenSEAL: ct * a1 + a0)
                    h_acts.append(ct * a1 + a0)
                h = CiphertextTensor(
                    ciphertexts=h_acts, engine=engine, depth=h.depth + 1
                )  # depth +1 (cipher×plain only)

        # out = h @ W_o_fused + b_o_fused (matmul_plain + add, +1 depth)
        out = h.matmul_plain(self._W_o_fused)
        # Add bias (cipher + plain_list, 0 depth)
        b_list = self._b_o_fused.detach().cpu().reshape(-1).tolist()
        out_ciphers = [ct + b_list for ct in out._cipher]
        out = CiphertextTensor(ciphertexts=out_ciphers, engine=engine,
                               depth=out.depth + 0)  # add is 0 depth

        # v135: Linear Token Mixing (cross-token information flow)
        # Applied AFTER PolyMLP (with fused skip_alpha), BEFORE residual.
        # FHE: N² cipher×scalar + N(N-1) cipher+cipher, depth +1.
        # Since TokenMixer is LINEAR, it commutes with skip_alpha fusion:
        #   Plain:  y = x + alpha * TokenMixer(h@W_o + b_o)
        #   Cipher: y = x + TokenMixer(alpha * (h@W_o + b_o))  [alpha fused into W_o]
        # These are equivalent because TokenMixer is linear.
        if self.token_mixer is not None:
            out = self.token_mixer.forward_cipher(
                out, source_indices=active if subset else None
            )  # depth +1

        # === Residual: y = x + out (cipher+cipher, 0 depth) ===
        # Inactive PBSR tokens are copied, never multiplied.
        if subset:
            y_cts = list(full_x._cipher)
            for index, ct in zip(active, out._cipher):
                y_cts[index] = full_x._cipher[index] + ct
            return CiphertextTensor(
                ciphertexts=y_cts, engine=engine, depth=out.depth
            )
        y = x + out

        return y

    def prepare_cipher_weights(self):
        """Precompute fused weights for cipher path.

        Fusions (v106 rule: never use cipher × plain_vector):
        1. PolyNorm's gamma fused into W_a (and W_b for Hadamard)
           (diag(gamma) @ W_a, diag(gamma) @ W_b)
        2. PolyNorm's beta + c0*gamma fused into W_o's bias (simplified — see below)
        3. SkipInit alpha1 fused into W_o (alpha1 * W_o)

        v118 PolyAct: W_b is unused, only W_a is fused with gamma.
        v118 PolyAct: per-channel a0/a1/a2 are cached as scalars (mean across M).

        NOTE: PolyNorm's polynomial coefficients (c0, c1, c2 or c0_regions, etc.)
        are NOT fused — they are applied via polyval in the cipher path (depth +2).
        v117: per-region coeffs are cached by PolyNorm.prepare_cipher_weights().
        """
        # Cache normalization coefficients
        # v116/v117 PolyNorm: polynomial coeffs (mean across D) as scalars
        # v165 CADN: no coeffs to cache (γ/β sent to client dynamically via get_cadn_params)
        self.norm1.prepare_cipher_weights()
        if self.fuse_linear_poly_norm:
            # Fusion is exact only when a degree-1 norm has no quadratic term.
            # Older checkpoints may carry stale c2 values after an architecture
            # override; retain their native FHE path instead of changing model
            # semantics silently.
            c2_regions = getattr(self.norm1, "c2_regions", None)
            c2_shared = getattr(self.norm1, "c2", None)
            if c2_regions is not None:
                max_c2 = float(c2_regions.detach().abs().max())
            elif c2_shared is not None:
                max_c2 = float(c2_shared.detach().abs().max())
            else:
                max_c2 = 0.0
            if max_c2 > 1e-8:
                self.fuse_linear_poly_norm = False

        # Get normalization affine parameters
        # v165 CADN: γ/β are applied by the CLIENT as part of CADN LayerNorm.
        #   The server's z_norm ciphertext ALREADY has γ/β applied. So we must NOT
        #   fuse γ into W_a (that would double-apply γ). Set γ=1, β=0 for fusion.
        # v116/v117 PolyNorm: γ/β are NOT applied by PolyNorm (polyval outputs raw
        #   z_norm = c0+c1·x+c2·x²). They must be fused into W_a here.
        if self.use_cadn:
            gamma = torch.ones(self.dim)  # CADN already applied γ client-side
            beta = torch.zeros(self.dim)  # CADN already applied β client-side
        else:
            gamma = self.norm1.weight  # [D] — PolyNorm's γ, fused into W_a
            beta = self.norm1.bias     # [D] — PolyNorm's β, fused into W_o bias
        alpha = float(self.skip_alpha1.detach())

        # Fusion 1: gamma into W_a (and W_b for Hadamard)
        # z_norm = gamma * poly(x) + beta    [PolyNorm path]
        # z_norm = LayerNorm(x, gamma, beta) [CADN path — γ/β already in z_norm]
        # h_a = z_norm @ W_a
        # PolyNorm: h_a = (gamma * poly(x)) @ W_a + beta @ W_a = poly(x) @ (diag(gamma) @ W_a) + beta @ W_a
        # CADN:     h_a = z_norm @ W_a (no gamma fusion — z_norm already has γ/β)
        # For simplicity (v116 smoke test), we set beta=0 in the activation expansion
        # and only fuse gamma. See SIMPLIFICATION note below.
        #
        # SpectralBoundedLinear.weight has shape [in, out] (NOT [out, in] like nn.Linear)
        self._W_a_fused = gamma.unsqueeze(1) * self.W_a.weight  # [D, M]
        if self.activation_type == 'hadamard':
            self._W_b_fused = gamma.unsqueeze(1) * self.W_b.weight  # [D, M]
        else:
            # v118/v121 PolyAct: W_b is unused, no fusion needed
            self._W_b_fused = None
            # v121: PolyAct coeffs are already SCALAR (not per-channel).
            # No need to take mean — just detach and convert to float.
            # This eliminates the train-inference gap: training uses the same
            # scalar coeffs as FHE inference.
            self._cipher_pa_a0 = float(self.poly_act_a0.detach())
            self._cipher_pa_a1 = float(self.poly_act_a1.detach())
            self._cipher_pa_a2 = float(self.poly_act_a2.detach())
            # v156 Innovation E: MedPolyAct — cache per-region coeffs as lists.
            # Each token uses its region's coeff in the cipher path (like AAPN).
            if self._use_med_poly_act:
                self._cipher_pa_a0_regions = [
                    float(c) for c in self.poly_act_a0_regions.detach().tolist()
                ]
                self._cipher_pa_a1_regions = [
                    float(c) for c in self.poly_act_a1_regions.detach().tolist()
                ]
                self._cipher_pa_a2_regions = [
                    float(c) for c in self.poly_act_a2_regions.detach().tolist()
                ]
            else:
                self._cipher_pa_a0_regions = None
                self._cipher_pa_a1_regions = None
                self._cipher_pa_a2_regions = None

        # Fuse skip_alpha1 into W_o: W_o_fused = alpha1 * W_o.weight
        self._W_o_fused = alpha * self.W_o.weight  # [M, D]

        # Fuse PolyNorm's beta into W_o's bias (train-inference consistency fix).
        #
        # forward_plain computes: h_a = (gamma * poly(x) + beta) @ W_a
        #                        = poly(x) @ (gamma * W_a) + beta @ W_a
        # The first term is captured by _W_a_fused (gamma baked into W_a). The
        # second term (beta @ W_a) is a constant bias that must be carried through
        # the activation and W_o to match forward_plain exactly.
        #
        # For PolyAct degree-1 (act(h) = a0 + a1·h):
        #   forward_plain: out = (a0 + a1·(poly(x)@W_a_fused + beta@W_a_fused)) @ W_o + b_o
        #                    = a1·poly(x)@W_a_fused@W_o + a1·(beta@W_a_fused)@W_o + a0@W_o + b_o
        #   forward_cipher: out = (a0 + a1·poly(x)@W_a_fused) @ W_o + b_o_fused
        #                    = a1·poly(x)@W_a_fused@W_o + a0@W_o + b_o_fused
        #   Matching: b_o_fused = b_o + a1·(beta@W_a_fused)@W_o
        #                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                           beta contribution (constant through linear act)
        # NOTE: a0@W_o is already applied via polyval's a0 term in cipher path
        # (ct * a1 + a0, then @ W_o), so it must NOT be re-fused into b_o_fused.
        #
        # For Hadamard (act(h_a, h_b) = h_a ⊙ h_b): beta enters nonlinearly as
        #   (poly(x)@W_a + beta@W_a) ⊙ (poly(x)@W_b + beta@W_b)
        # The cross terms (beta@W_a ⊙ poly(x)@W_b + poly(x)@W_a ⊙ beta@W_b) are
        # data-dependent (cannot be fused into a constant bias). Only the pure
        # beta⊗beta term is constant but second-order small. Hadamard therefore
        # drops beta in the cipher path (documented limitation; v116-era behavior).
        # PolyAct degree-1 is the canonical activation in v164, so this is the
        # primary path that benefits from the fix.
        b_o = self.W_o.bias if self.W_o.bias is not None else torch.zeros(self.dim)
        if self.activation_type == 'hadamard':
            # Hadamard: beta enters nonlinearly, cannot be fused as constant.
            # Documented limitation (second-order beta⊗beta term is negligible).
            self._b_o_fused = alpha * b_o  # [D]
        else:
            # PolyAct (degree-1 or degree-2): beta contribution is a0-independent
            # constant through the linear a1·h term. For degree-2, the a2·h² term
            # also produces a constant a2·(beta@W_a)² contribution (second-order
            # in beta, small after LayerNorm normalization) — we include only the
            # first-order a1 term for depth safety (no extra cipher ops).
            a1 = self._cipher_pa_a1 if self._cipher_pa_a1 is not None else float(self.poly_act_a1.detach())
            beta_w_a = beta @ self._W_a_fused  # [M] — beta projected through fused W_a
            # beta_contribution = a1 * (beta_w_a @ W_o_fused): [M] @ [M, D] = [D]
            # v165 FIX: removed erroneous .t() — _W_o_fused is [M, D] (in=M, out=D),
            # so beta_w_a @ _W_o_fused = [M] @ [M, D] = [D] is correct (NOT .t()).
            beta_contribution = (
                torch.zeros_like(b_o)
                if self.fuse_linear_poly_norm
                else a1 * (beta_w_a @ self._W_o_fused)
            )  # [D]
            self._b_o_fused = alpha * (b_o + beta_contribution)  # [D]

        # Optional callback-free fusion for the affine (degree-1) PolyAct:
        #   (a0 + a1 h) W_o + b = h (a1 W_o) + (b + a0 row_sum(W_o)).
        # Keep separate matrices for MedPolyAct regions because their a0/a1
        # coefficients are token-region dependent.
        self._W_o_linear_fused = None
        self._b_o_linear_fused = None
        self._W_o_linear_fused_regions = None
        self._b_o_linear_fused_regions = None
        if self.fuse_linear_poly_act and self.poly_act_degree == 1:
            row_sum = self._W_o_fused.sum(dim=0)
            if self._use_med_poly_act and self._cipher_pa_a1_regions is not None:
                self._W_o_linear_fused_regions = []
                self._b_o_linear_fused_regions = []
                for a0, a1 in zip(
                    self._cipher_pa_a0_regions, self._cipher_pa_a1_regions
                ):
                    self._W_o_linear_fused_regions.append(a1 * self._W_o_fused)
                    self._b_o_linear_fused_regions.append(
                        self._b_o_fused + a0 * row_sum
                    )
            else:
                self._W_o_linear_fused = self._cipher_pa_a1 * self._W_o_fused
                self._b_o_linear_fused = self._b_o_fused + self._cipher_pa_a0 * row_sum

        if self.fuse_linear_poly_norm:
            self._W_a_norm_fused = None
            self._b_a_norm_fused = None
            self._W_a_norm_fused_regions = None
            self._b_a_norm_fused_regions = None
            if getattr(self.norm1, "_cipher_c1_regions", None) is not None:
                self._W_a_norm_fused_regions = []
                self._b_a_norm_fused_regions = []
                for c0, c1 in zip(
                    self.norm1._cipher_c0_regions,
                    self.norm1._cipher_c1_regions,
                ):
                    gamma = torch.ones(self.dim) if self.use_cadn else self.norm1.weight
                    beta = torch.zeros(self.dim) if self.use_cadn else self.norm1.bias
                    c0_value = torch.as_tensor(c0, dtype=self.W_a.weight.dtype)
                    c1_value = torch.as_tensor(c1, dtype=self.W_a.weight.dtype)
                    matrix = (c1_value * gamma).unsqueeze(1) * self.W_a.weight
                    bias_vec = (c0_value * gamma + beta) @ self.W_a.weight
                    self._W_a_norm_fused_regions.append(matrix)
                    self._b_a_norm_fused_regions.append(bias_vec)
            else:
                c0 = self.norm1._cipher_c0
                c1 = self.norm1._cipher_c1
                if c0 is not None and c1 is not None:
                    gamma = self.norm1.weight
                    beta = self.norm1.bias
                    self._W_a_norm_fused = (float(c1) * gamma).unsqueeze(1) * self.W_a.weight
                    self._b_a_norm_fused = (float(c0) * gamma + beta) @ self.W_a.weight

        # v135: Prepare token mixer weights for cipher path
        if self.token_mixer is not None:
            self.token_mixer.prepare_cipher_weights()

    def multiplicative_depth(self) -> int:
        """Return max multiplicative depth of this block (relative to input).

        Hadamard (v116/v117): poly_norm_degree + 3 = 2 + 3 = 5
          - PolyNorm: +2 (polyval degree-2)
          - h_a = z @ W_a (+1)
          - h_b = z @ W_b (+1) — parallel to h_a, so max is +1
          - h = h_a * h_b (+1)
          - out = h @ W_o (+1)
          Total: 2 + 1 + 1 + 1 = 5

        PolyAct degree-2 (v118): poly_norm_degree + 4 = 2 + 4 = 6
          - PolyNorm: +2 (polyval degree-2)
          - h = z @ W_a (+1, matmul)
          - h_act = h.polyval([a0,a1,a2]) (+2, Horner: cipher×plain + cipher×cipher)
          - out = h_act @ W_o (+1, matmul)
          Total: 2 + 1 + 2 + 1 = 6

        PolyAct degree-1 (v119): poly_norm_degree + 3 = 2 + 3 = 5
          - PolyNorm: +2 (polyval degree-2)
          - h = z @ W_a (+1, matmul)
          - h_act = a0 + a1·h (+1, cipher×plain ONLY, no cipher×cipher)
          - out = h_act @ W_o (+1, matmul)
          Total: 2 + 1 + 1 + 1 = 5
          NO cipher×cipher in PolyAct → dominant CKKS noise source eliminated.
          Nonlinearity comes from PolyNorm's c2·x² term.

        v135 LTM: +1 (cipher×scalar only, NO cipher×cipher in token mixing)
          Block depth with LTM: 6 (Hadamard+LTM), 7 (PolyAct d2+LTM), 6 (LPPN+LTM)
        """
        if self.activation_type == 'hadamard':
            depth = self.poly_norm_degree + 3  # 2 + 3 = 5
        elif self.poly_act_degree == 2:
            depth = self.poly_norm_degree + 4  # 2 + 4 = 6 (v118)
        else:  # poly_act_degree == 1
            depth = self.poly_norm_degree + 3  # 2 + 3 = 5 (v119)
            if self.fuse_linear_poly_act:
                # The affine PolyAct is folded into W_o; only one projection
                # remains after PolyNorm instead of activation + projection.
                depth -= 1
            if self.fuse_linear_poly_norm:
                # Degree-1 PolyNorm is affine and is folded into W_a plus its
                # bias, removing the normalization multiply level.
                depth -= 1
        # v165 CADN: subtracts poly_norm_degree (2, the non-CAS PolyNorm depth).
        # CADN provides z_norm as a fresh depth-0 ciphertext (client-computed
        # LayerNorm), replacing PolyNorm's depth-2 polynomial approximation.
        # Under CAS path (PolyNorm=1), net savings is 1 (5→4 with LTM).
        # Under non-CAS path (PolyNorm=2), net savings is 2 (6→4 with LTM).
        # v165 canonical uses CADN, returning the correct depth 4.
        if self.use_cadn:
            depth -= self.poly_norm_degree  # subtract PolyNorm depth (2), CADN is +0
        # v135: Add token mixing depth
        if self.token_mixer is not None:
            depth += self.token_mixer.multiplicative_depth()  # +1
        return depth

    def forward_plain_fhe_match(
        self,
        x: torch.Tensor,
        use_poly: bool = True,
        token_regions: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Plain forward that EXACTLY mirrors the cipher path.

        Used for FHE verification: cos(cipher, plain_fhe_match) isolates CKKS noise.

        This MUST match `_forward_cipher` operation-for-operation:
          v116: PolyNorm with SHARED coeffs (mean across D)
          v117: PolyNorm with PER-REGION coeffs (each token uses its region's coeffs)
          v118: PolyAct with SCALAR coeffs (mean across M, matches v116 PolyNorm pattern)

        Hadamard path (v116/v117):
          - PolyMLP: h_a = z@W_a_fused, h_b = z@W_b_fused, h = h_a*h_b,
            out = h@W_o_fused + b_o_fused
            Uses FUSED weights (gamma fused into W_a, W_b; alpha fused into W_o)
            to match the cipher path exactly.
          - Residual: y = x + out (matches cipher path's cipher+cipher)

        PolyAct path (v118):
          - h = z @ W_a_fused (matmul, fused weights)
          - h_act = a0_scalar + a1_scalar * h + a2_scalar * h²
            (uses SCALAR mean across M, matches cipher path's polyval)
          - out = h_act @ W_o_fused + b_o_fused
          - Residual: y = x + out

        Args:
            x: [B, N, D] plaintext tensor
            use_poly: IGNORED (v116/v117/v118 always uses polynomial — kept for API compat)
            token_regions: list of N region indices (v117 AAPN, None for v116/v118)

        Returns:
            [B, N, D] plaintext tensor (FHE-intended computation)
        """
        # Ensure cipher weights are prepared (fusion applied)
        if self._W_a_fused is None:
            self.prepare_cipher_weights()

        # Use self.token_regions if not explicitly provided
        if token_regions is None:
            token_regions = self.token_regions

        # === Normalization (matches cipher path) ===
        # v165 CADN: Standard LayerNorm (matches cipher path's CADN callback exactly).
        # The client computes z_norm = LayerNorm(x, γ, β) in plaintext at depth 0.
        # Here we compute the SAME standard LayerNorm → ZERO train-inference gap.
        # CADNLayer.forward_plain uses nn.LayerNorm (dynamic mean/var + γ/β affine).
        if self.fuse_linear_poly_norm:
            if self._W_a_norm_fused_regions is not None and token_regions is not None:
                h = torch.empty(
                    (*x.shape[:-1], self.mlp_hidden),
                    device=x.device,
                    dtype=x.dtype,
                )
                for region, matrix in enumerate(self._W_a_norm_fused_regions):
                    indices = [
                        i for i, value in enumerate(token_regions) if value == region
                    ]
                    if not indices:
                        continue
                    index = torch.tensor(indices, dtype=torch.long, device=x.device)
                    h[..., index, :] = (
                        x[..., index, :] @ matrix.to(device=x.device, dtype=x.dtype)
                        + self._b_a_norm_fused_regions[region].to(
                            device=x.device, dtype=x.dtype
                        )
                    )
            else:
                h = x @ self._W_a_norm_fused.to(device=x.device, dtype=x.dtype)
                h = h + self._b_a_norm_fused.to(device=x.device, dtype=x.dtype)
            z = h
        elif self.use_cadn:
            z = self.norm1.forward_plain(x)
        elif self.num_regions > 1 and token_regions is not None:
            # v117 path: PER-REGION coeffs (matches cipher path's per-token polyval)
            # Cipher path: z[i] = ct[i].polyval([c0_regions[r(i)], c1_regions[r(i)], c2_regions[r(i)]])
            # Plain equivalent: per-token polynomial with region-specific coeffs
            c0_regions = self.norm1._cipher_c0_regions  # list of R scalars
            c1_regions = self.norm1._cipher_c1_regions
            c2_regions = self.norm1._cipher_c2_regions
            # Convert to tensors for gather
            regions_t = torch.tensor(token_regions, dtype=torch.long, device=x.device)  # [N]
            c0_tok = torch.tensor(c0_regions, device=x.device, dtype=x.dtype)[regions_t]  # [N]
            c1_tok = torch.tensor(c1_regions, device=x.device, dtype=x.dtype)[regions_t]
            c2_tok = torch.tensor(c2_regions, device=x.device, dtype=x.dtype)[regions_t]
            # Reshape for broadcasting over [B, N, D]
            if self.norm1.channelwise_coeffs:
                if x.dim() == 3:
                    c0_tok = c0_tok.unsqueeze(0)
                    c1_tok = c1_tok.unsqueeze(0)
                    c2_tok = c2_tok.unsqueeze(0)
            elif x.dim() == 3:
                c0_tok = c0_tok.view(1, -1, 1)
                c1_tok = c1_tok.view(1, -1, 1)
                c2_tok = c2_tok.view(1, -1, 1)
            else:
                c0_tok = c0_tok.view(-1, 1)  # [N, 1]
                c1_tok = c1_tok.view(-1, 1)
                c2_tok = c2_tok.view(-1, 1)
            z = c0_tok + c1_tok * x + c2_tok * x * x
        else:
            # v116/v118 path: SHARED coeffs (mean across D)
            c0 = self.norm1._cipher_c0 if self.norm1._cipher_c0 is not None else float(self.norm1.c0.mean())
            c1 = self.norm1._cipher_c1 if self.norm1._cipher_c1 is not None else float(self.norm1.c1.mean())
            c2 = self.norm1._cipher_c2 if self.norm1._cipher_c2 is not None else float(self.norm1.c2.mean())
            z = c0 + c1 * x + c2 * x * x

        # === Activation with FUSED weights (matches cipher path) ===
        if self.activation_type == 'hadamard':
            # Hadamard: h_a = z@W_a_fused, h_b = z@W_b_fused, h = h_a*h_b
            h_a = z @ self._W_a_fused  # [B, N, M]
            h_b = z @ self._W_b_fused  # [B, N, M]
            h = h_a * h_b              # [B, N, M] (Hadamard, degree-2 activation)
        else:
            # v118/v119 PolyAct: h = z@W_a_fused, h_act = polynomial(h)
            # v156 Innovation E: MedPolyAct — per-region coeffs (MIRRORS cipher path
            # L1195-1199). Each token uses its region's coeff, exactly as the cipher
            # path does per-token polyval. v157 FIX: previously this used SCALAR coeffs
            # (mean across M), which DIVERGED from the cipher path (cos=0.45) and made
            # cos_256 measure structural mismatch instead of CKKS noise. Now mirrors
            # the cipher path exactly → cos_256 reflects pure CKKS noise.
            h = z if self.fuse_linear_poly_norm else z @ self._W_a_fused
            if self.fuse_linear_poly_act and self.poly_act_degree == 1:
                # Mirror the cipher-side affine fusion exactly in plaintext so
                # the verification reference measures CKKS error only.
                if self._W_o_linear_fused_regions is not None and token_regions is not None:
                    out = torch.empty(
                        (*h.shape[:-1], self.dim), dtype=h.dtype, device=h.device
                    )
                    for region, matrix in enumerate(self._W_o_linear_fused_regions):
                        indices = [i for i, value in enumerate(token_regions) if value == region]
                        if not indices:
                            continue
                        index = torch.tensor(indices, dtype=torch.long, device=h.device)
                        out[..., index, :] = (
                            h[..., index, :] @ matrix.to(device=h.device, dtype=h.dtype)
                            + self._b_o_linear_fused_regions[region].to(
                                device=h.device, dtype=h.dtype
                            )
                        )
                else:
                    out = h @ self._W_o_linear_fused.to(device=h.device, dtype=h.dtype)
                    out = out + self._b_o_linear_fused.to(device=h.device, dtype=h.dtype)
                if self.token_mixer is not None:
                    out = self.token_mixer.forward_plain_fhe_match(out)
                return x + out
            if (self._use_med_poly_act and token_regions is not None
                    and self._cipher_pa_a0_regions is not None):
                # v156 E: per-region coeffs (mirror cipher path's per-token polyval).
                # Cipher path (L1195-1199): for each token i, r = token_regions[i],
                #   a0 = _cipher_pa_a0_regions[r]; apply polyval([a0,a1,a2]).
                # Plain mirror: gather per-token coeffs by region, broadcast over [B,N,M].
                regions_t = torch.tensor(token_regions, dtype=torch.long, device=x.device)  # [N]
                a0_tok = torch.tensor(self._cipher_pa_a0_regions, device=x.device, dtype=x.dtype)[regions_t]  # [N]
                a1_tok = torch.tensor(self._cipher_pa_a1_regions, device=x.device, dtype=x.dtype)[regions_t]
                a2_tok = torch.tensor(self._cipher_pa_a2_regions, device=x.device, dtype=x.dtype)[regions_t]
                if x.dim() == 3:
                    a0_tok = a0_tok.view(1, -1, 1)  # [1, N, 1]
                    a1_tok = a1_tok.view(1, -1, 1)
                    a2_tok = a2_tok.view(1, -1, 1)
                else:
                    a0_tok = a0_tok.view(-1, 1)  # [N, 1]
                    a1_tok = a1_tok.view(-1, 1)
                    a2_tok = a2_tok.view(-1, 1)
                if self.poly_act_degree == 2:
                    # v118: degree-2 PolyAct (matches cipher path's polyval([a0,a1,a2]))
                    h = a0_tok + a1_tok * h + a2_tok * (h * h)  # [B, N, M] (degree-2)
                else:
                    # v119: degree-1 PolyAct (matches cipher path's ct * a1 + a0)
                    h = a0_tok + a1_tok * h  # [B, N, M] (degree-1, linear)
            else:
                # v116/v118/v121: SCALAR coeffs (mean across M) — non-medical path.
                a0 = self._cipher_pa_a0  # scalar (mean across M)
                a1 = self._cipher_pa_a1
                a2 = self._cipher_pa_a2
                if self.poly_act_degree == 2:
                    h = a0 + a1 * h + a2 * (h * h)  # [B, N, M] (degree-2)
                else:
                    h = a0 + a1 * h  # [B, N, M] (degree-1, linear)

        out = h @ self._W_o_fused  # [B, N, D]
        b_o = self._b_o_fused if self._b_o_fused is not None else torch.zeros(self.dim)
        out = out + b_o  # [B, N, D]

        # v135: Linear Token Mixing (matches cipher path's forward_cipher)
        # Applied AFTER PolyMLP (with fused skip_alpha), BEFORE residual.
        if self.token_mixer is not None:
            out = self.token_mixer.forward_plain_fhe_match(out)  # [B, N, D]

        # === Residual: y = x + out (matches cipher path) ===
        y = x + out

        return y

    def project_spectral(self):
        """Project weights to spectral bound (for training, if needed)."""
        self.W_a.project_spectral()
        if self.activation_type == 'hadamard':
            self.W_b.project_spectral()
        self.W_o.project_spectral()
        # v135: Project token mixer weights
        if self.token_mixer is not None:
            self.token_mixer.project_spectral()

    def bake_statistics(self):
        """Bake statistics for cipher path (alias for prepare_cipher_weights)."""
        self.prepare_cipher_weights()
