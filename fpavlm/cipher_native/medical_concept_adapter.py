"""MedicalConceptAdapter: Plug-and-Play LLM Interface for FHE ViT (v129 PCSD).

v129 INNOVATION: Privacy-Computation Splitting for Non-Linear Decoders (PCSD).
v128 INNOVATION: Client-Side Spatial Broadcast (CSB) — Privacy-Computation Splitting

FIRST-PRINCIPLES ANALYSIS (why v124/v125/v126 TF-IDF=1.0 is semantically vacuous):

  MedGemma's LLM expects 256 image tokens in a 16×16 grid (SigLIP native after
  avg_pool kernel=4). Each token carries UNIQUE spatial information about a
  specific region of the original 896×896 image. The LLM's attention pattern
  is trained on this 256-token spatial layout.

  v124 (global pool → 1 token): 256 positions filled with 1 value → no spatial
  v125 (region pool → 5 tokens): 5 spatial regions, but LLM expects 256
  v126 (5 tokens repeated → 256): 256 positions cycling through 5 values →
    NO spatial diversity (verified analytically — rank ≤ 5)

  ROOT CAUSE: The LLM detects "no spatial diversity" in image tokens and falls
  back to text-only prior ("no findings"). This is NOT an FHE problem — it's a
  TOKEN COUNT + SPATIAL DIVERSITY mismatch.

v127 → v128 EVOLUTION (first-principles discovery of v127 architectural flaw):

  v127 SBA (Server-side Bilinear interp): projected 16 ciphers then performed
  bilinear interpolation UNDER FHE (256×4 cipher×scalar + 768 cipher+cipher ops
  per image). Empirically this made each image take 60+ minutes — UNACCEPTABLE.

  FIRST-PRINCIPLES INSIGHT: The bilinear interpolation weights are PUBLIC and
  DETERMINISTIC (derived only from grid sizes 4 and 16). They contain NO model
  parameters and NO private information. Performing public operations under FHE
  wastes cipher ops with ZERO privacy benefit — the same protection is achieved
  by doing them AFTER decryption, in plaintext on the client.

  Privacy-Computation Splitting principle (v128's core contribution):
    Operations that involve PRIVATE model weights (W_concept) → DO UNDER FHE
    Operations that use PUBLIC transforms (bilinear interp) → DO IN PLAINTEXT
    The client only needs to decrypt 16 ciphers + apply a public 256×16 matrix.
    This is trivial client compute (a single plaintext matmul on 16 tokens).

v128 CSB PIPELINE:
  Server (FHE):  16 input ciphers → 16 projected ciphers (16 matmuls, depth +1)
                  ↓ transmit 16 ciphers to client
  Client (plain): decrypt 16 ciphers → 16 plaintext tokens [16, llm_dim]
                  bilinear interp 4×4→16×16 (public W_up [256,16] @ [16,llm_dim])
                  → 256 plaintext tokens [256, llm_dim] (matches SigLIP native)
                  feed to LLM

v128 vs v127 SBA comparison (BOTH produce 256 spatially diverse tokens):
  Metric                  | v127 SBA (server-side) | v128 CSB (client-side)
  ------------------------+------------------------+----------------------
  Cipher ops per image    | 16 matmuls + ~1800 ops | 16 matmuls ONLY
  FHE multiplicative depth| 2 (matmul + scalar)    | 1 (matmul only)
  Cipher ops × image time | 60+ minutes (FAILED)   | ~5 minutes (v126-like)
  Spatial diversity       | 256 unique tokens      | 256 unique tokens (SAME)
  Privacy (server→client) | Server sees ciphers    | Server sees ciphers (SAME)
  Privacy (client→server) | Client sees 256 plains | Client sees 16 plains
  Accuracy (cos FHE/plain)| identical              | identical (linear ops)
  Client compute          | decrypt 256 ciphers    | decrypt 16 + matmul

  v128 is STRICTLY BETTER than v127 SBA on every metric except client-side
  cipher count (which is irrelevant since client only decrypts). The 16 vs 256
  cipher transmission is a 16× bandwidth win for v128 too.

ALGEBRAIC EQUIVALENCE PROOF (v127 SBA cipher path == v128 CSB):
  Let P_i = cipher_i.matmul(W_concept) + b  (projected cipher, i ∈ [0..15])
  Let W_up ∈ R^{256×16} be the bilinear interp matrix (PUBLIC, fixed)

  v127 SBA output token k (under FHE):
    out_k = Σ_i W_up[k,i] · P_i           (cipher × scalar + cipher + cipher)

  v128 CSB output token k (after client-side interp):
    plain_i = decrypt(P_i)                 (decryption is LINEAR)
    out_k = Σ_i W_up[k,i] · plain_i
         = Σ_i W_up[k,i] · decrypt(P_i)
         = decrypt(Σ_i W_up[k,i] · P_i)   (linearity of CKKS decryption)
         = decrypt(v127 out_k)             (IDENTICAL to v127)

  So v127 and v128 produce IDENTICAL plaintext outputs (up to CKKS noise).
  v128 just moves the bilinear interp from FHE-domain to plain-domain.
  This is FUNDAMENTALLY more efficient with NO accuracy cost.

v124/v125/v126/v127/v128/v129 comparison:
  v124 (global pool):       1 token,  no spatial, TF-IDF(FHE,GT)=0.03-0.15
  v125 (region pool):       5 tokens, some spatial, TF-IDF=0.84-1.0 (still vacuous)
  v126 (SigLIP distill):    5 tokens, in-distribution, TF-IDF=1.0 (STILL vacuous)
  v127 (SBA, server-side):  256 tokens, but 60+ min/image — UNUSABLE
  v128 (CSB, client-side):  256 tokens, ~5 min/image, BUT TF-IDF=1.0 (rank-16)
  v129 (PCSD, client-side): 256 tokens, ~5 min/image, rank-256 via Perceiver
    ← FIXES v128's rank-16 bottleneck with NON-LINEAR cross-attention decoder

v129 RANK BOTTLENECK ANALYSIS (first-principles):

  v128 CSB uses BILINEAR INTERPOLATION (linear): out = W_up @ P, where
  W_up ∈ R^{256×16}, P ∈ R^{16×D}. By rank inequality:
    rank(out) ≤ min(rank(W_up), rank(P)) ≤ min(16, 16) = 16

  So v128's 256 output tokens span AT MOST 16 dimensions in 2560-dim space.
  The LLM detects this rank-16 deficit (via low-rank attention pattern) and
  falls back to text-only prior, producing IDENTICAL reports (TF-IDF=1.0).

  v129 PCSD uses Perceiver cross-attention (non-linear). The softmax in
  attention breaks the rank-16 bottleneck: output rank can reach 256 (full
  rank in 2560-dim space). The LLM sees spatially diverse tokens and
  generates pathology-specific reports.

FHE depth comparison:
  v124: sum (0) + scale (+1) + matmul (+1) = 2
  v125: per-region sum (0) + scale (+1) + matmul (+1) = 2
  v128: matmul (+1) ONLY = 1   ← DEPTH REDUCTION (best of all versions!)
  v129: matmul (+1) ONLY = 1   ← SAME as v128 (decoder is CLIENT-SIDE, +0 depth)

Privacy (v129 PCSD, SAME as v128 CSB):
  - Server: holds W_concept (PRIVATE), does projection under FHE (16 matmuls)
  - Client: decrypts 16 ciphers, applies PUBLIC Perceiver decoder, feeds to LLM
  - Server NEVER sees plaintext features
  - Client NEVER sees W_concept (only sees the projected encrypted output)
  - Perceiver decoder weights are PUBLIC (shared with client) — does NOT leak
    the FHE encoder weights (decoder is applied AFTER decryption, on plaintext)

Medical concept tokens (v124-v126 legacy, NOT used in v128 CSB / v129 PCSD):
  - 32 learnable tokens representing medical concepts (plaintext, client-side)
  - v128/v129 produce 256 tokens matching SigLIP native count, so concept
    tokens are NOT concatenated (would make total 288, OOD for LLM)
  - Kept for backward compatibility with v124/v125/v126 experiments
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# v167: Plug-and-play perceiver registry — allow swapping perceiver by name
from .base_perceiver import BasePerceiverDecoder, PerceiverRegistry
from .ciphertext_tensor import CiphertextTensor
from .diagnostic_bottleneck import DiagnosticBottleneckRelease
from .perceiver_decoder import PerceiverDecoder
from .spectral_bounded import SpectralBoundedLinear


def compute_bilinear_upsample_matrix(
    student_grid: int = 4,
    target_grid: int = 16,
) -> torch.Tensor:
    """Compute bilinear interpolation matrix for spatial upsampling.

    For upsampling from student_grid×student_grid to target_grid×target_grid,
    each output position (i, j) in target_grid is a weighted combination of
    up to 4 input positions in student_grid (fewer at edges due to clamping).

    Uses ACCUMULATION to correctly handle edge cases where input indices
    coincide (e.g., when x0==x1 due to clamping at student_grid-1).

    Args:
        student_grid: Input grid size (default 4 → 16 tokens)
        target_grid: Output grid size (default 16 → 256 tokens)

    Returns:
        W_up: [target_grid^2, student_grid^2] sparse matrix. Each row has
              1-4 non-zero entries that sum to 1 (bilinear interp property).

    Mathematical formulation (align_corners=False, matches F.interpolate):
      For output position (i, j) in target_grid:
        src_x = (i + 0.5) * student_grid / target_grid - 0.5
        src_y = (j + 0.5) * student_grid / target_grid - 0.5
        src_x = clamp(src_x, 0, student_grid - 1)
        src_y = clamp(src_y, 0, student_grid - 1)
        x0 = floor(src_x), x1 = min(x0 + 1, student_grid - 1), dx = src_x - x0
        y0 = floor(src_y), y1 = min(y0 + 1, student_grid - 1), dy = src_y - y0
        weights: w00 = (1-dx)*(1-dy), w10 = dx*(1-dy), w01 = (1-dx)*dy, w11 = dx*dy
        output(i, j) = w00*input(x0,y0) + w10*input(x1,y0) + w01*input(x0,y1) + w11*input(x1,y1)

    Edge case handling: When src_x or src_y is clamped (at the boundary),
    x0 may equal x1 (or y0 may equal y1), causing in00==in10 (or in00==in01).
    We ACCUMULATE the weights at the coinciding index (e.g., w00+w10 at in00).
    This ensures each row sums to exactly 1.0.
    """
    W_up = torch.zeros(target_grid * target_grid, student_grid * student_grid)
    for i in range(target_grid):
        for j in range(target_grid):
            # Source position in student_grid (align_corners=False)
            src_x = (i + 0.5) * student_grid / target_grid - 0.5
            src_y = (j + 0.5) * student_grid / target_grid - 0.5
            src_x = max(0.0, min(student_grid - 1.0, src_x))
            src_y = max(0.0, min(student_grid - 1.0, src_y))
            x0 = int(math.floor(src_x))
            x1 = min(x0 + 1, student_grid - 1)
            dx = src_x - x0
            y0 = int(math.floor(src_y))
            y1 = min(y0 + 1, student_grid - 1)
            dy = src_y - y0
            # Bilinear weights
            w00 = (1.0 - dx) * (1.0 - dy)
            w10 = dx * (1.0 - dy)
            w01 = (1.0 - dx) * dy
            w11 = dx * dy
            # Output index
            out_idx = i * target_grid + j
            # Input indices (in student_grid×student_grid flattened, row-major)
            in00 = x0 * student_grid + y0
            in10 = x1 * student_grid + y0
            in01 = x0 * student_grid + y1
            in11 = x1 * student_grid + y1
            # ACCUMULATE weights (handles edge cases where indices coincide)
            W_up[out_idx, in00] += w00
            W_up[out_idx, in10] += w10
            W_up[out_idx, in01] += w01
            W_up[out_idx, in11] += w11
    return W_up


class MedicalConceptAdapter(nn.Module):
    """Plug-and-play LLM interface for FHE ViT (v129 PCSD / v128 CSB).

    Converts encrypted ViT features to LLM-compatible embeddings via:
      1. Pooling/upsampling:
         - 'global' (v124): pool N tokens → 1 token
         - 'region' (v125/v126): pool N tokens → K region tokens
         - 'spatial_upsample' (v128 CSB): upsample N=16 tokens (4×4 grid)
           → 256 tokens (16×16 grid, matches SigLIP native count).
           v128 CSB: SERVER projects 16 ciphers (matmul only, depth 1);
           CLIENT decrypts 16 ciphers and applies PUBLIC bilinear interp
           to produce 256 plaintext tokens.
         - 'perceiver_decoder' (v129 PCSD): cross-attention decoder
           16 input tokens → 256 output tokens via Perceiver-style attention.
           v129 PCSD: SERVER projects 16 ciphers (matmul only, depth 1, SAME
           as v128); CLIENT decrypts 16 ciphers and applies PUBLIC Perceiver
           decoder (NON-LINEAR) to produce 256 spatially diverse tokens.
           BREAKS v128's rank-16 bottleneck (linear → rank ≤ 256).
      2. Concept projection: project D-dim features to LLM-dim
      3. Concept token concatenation (v124-v126 legacy, NOT used in v128/v129)

    FHE depth:
      - global/region: pool (+1) + matmul (+1) = 2
      - spatial_upsample (v128 CSB): matmul (+1) ONLY = 1  ← BEST (depth reduction)
        (bilinear interp is done CLIENT-SIDE on plaintext, +0 FHE depth)
      - perceiver_decoder (v129 PCSD): matmul (+1) ONLY = 1  ← SAME as v128
        (Perceiver decoder is done CLIENT-SIDE on plaintext, +0 FHE depth)

    Args:
        dim: ViT feature dimension D
        llm_dim: LLM input dimension (e.g., 2560 for MedGemma 4B)
        num_concepts: Number of learnable medical concept tokens (default 32)
            NOTE: Only used for 'global'/'region' pool types (v124-v126 legacy).
            v128/v129/v143 do NOT concatenate concept tokens.
        num_tokens: Number of ViT tokens N (for pooling)
        pool_type: 'global', 'region', 'spatial_upsample', 'perceiver_decoder',
            or 'direct' (v144 no-decoder pipeline — student grid matches SigLIP
            native 16×16=256).
        num_regions: Number of anatomical regions K (default 5, for 'region' pool)
        student_grid: Student FHE image grid size (default 4 → 16 tokens)
            Used for 'spatial_upsample', 'perceiver_decoder', 'direct'.
        target_grid: Target LLM image grid size (default 16 → 256 tokens)
            Used for 'spatial_upsample', 'perceiver_decoder', 'direct'.
            Must match LLM's expected token count (256 for MedGemma SigLIP after
            avg_pool kernel=4).
        perceiver_config: Optional dict of PerceiverDecoder config overrides
            (only used for 'perceiver_decoder' pool_type). Keys: num_heads,
            attention_dim, ffn_dim, num_layers, init_from_bilinear.

    v144 'direct' pool_type (NO decoder — the optimal architecture):
      Student grid = target grid = 16 (both 16×16 = 256 tokens, matching SigLIP
      native resolution). NO pooling, NO decoder, NO upsampling. W_concept
      projects ALL 256 tokens directly: [B, 256, D] → [B, 256, llm_dim].
      - NO decoder → NO parameter sharing (satisfies "server doesn't share params")
      - NO rank bottleneck → rank = 256 (accuracy preserved, no 16→256 expansion)
      - Direct 1:1 distillation → student token i ↔ teacher token i
      - FHE depth: 1 (W_concept matmul only, SAME as v143)
      - Client compute: decrypt 256 tokens → feed to LLM (trivial, NO transform)
    """

    def __init__(
        self,
        dim: int,
        llm_dim: int = 3072,
        num_concepts: int = 32,
        num_tokens: int = 16,
        pool_type: str = "global",
        num_regions: int = 5,
        student_grid: int = 4,
        target_grid: int = 16,
        perceiver_config: Optional[dict] = None,
        anatomical_prior: Optional[torch.Tensor] = None,
        disease_prior_embeddings: Optional[torch.Tensor] = None,
        # v156 Innovation G: disease-anchor init for W_concept
        disease_anchor_directions: Optional[torch.Tensor] = None,
        disease_anchor_alpha: float = 0.3,
        # v156 Innovation H: MedICMC — per-region magnitude calibration
        region_salience: Optional[torch.Tensor] = None,
        # v167: Plug-and-play perceiver — select perceiver by registry name
        # Default "v162-default" = canonical PerceiverDecoder (backward compat)
        # "identity" = IdentityPerceiver (ablation: no-op, repeat 16×16=256)
        # Future: "anatomy-aware", "frequency-aware", etc. can be registered
        perceiver_name: str = "v162-default",
        # v167: Inject a pre-built perceiver (bypasses registry construction).
        # When provided, perceiver_name/perceiver_config/anatomical_prior/etc.
        # are ignored. Useful for externally constructed perceivers.
        perceiver_instance: Optional[BasePerceiverDecoder] = None,
    ):
        super().__init__()
        valid_pool_types = ("global", "region", "spatial_upsample",
                            "perceiver_decoder", "direct")
        if pool_type not in valid_pool_types:
            raise ValueError(
                f"pool_type must be one of {valid_pool_types}, got {pool_type!r}"
            )
        self.dim = dim
        self.llm_dim = llm_dim
        self.num_concepts = num_concepts
        self.num_tokens = num_tokens
        self.pool_type = pool_type
        self.num_regions = num_regions if pool_type == "region" else 1

        # v127 SBA / v128 CSB / v129 PCSD / v144 Direct: spatial params
        if pool_type in ("spatial_upsample", "perceiver_decoder", "direct"):
            self.student_grid = student_grid
            self.target_grid = target_grid
            # Validate: student_grid^2 must match num_tokens
            if student_grid * student_grid != num_tokens:
                raise ValueError(
                    f"student_grid^2 ({student_grid**2}) must match num_tokens "
                    f"({num_tokens}) for {pool_type!r} pool_type"
                )
        else:
            self.student_grid = None
            self.target_grid = None

        if pool_type == "direct":
            # v144: Direct projection — NO pooling, NO decoder, NO upsampling.
            # Student grid == target grid (both 16×16 = 256, matching SigLIP native).
            # W_concept projects ALL 256 tokens directly. NO rank bottleneck.
            # NO decoder → NO parameter sharing (constraint compliant).
            self.W_up = None
            self.perceiver_decoder = None
        elif pool_type == "spatial_upsample":
            # v128 CSB: Pre-compute bilinear interpolation matrix [target^2, student^2]
            # This matrix is PUBLIC (derived from grid sizes only — no model params).
            # v128 CSB: applied CLIENT-SIDE on plaintext (DEPRECATED;
            #   v129 PCSD's server_side_perceiver_decode is the canonical path).
            # v127 SBA used to apply it SERVER-SIDE under FHE, but that was 100× slower
            # with no privacy benefit (public weights don't need FHE protection).
            W_up = compute_bilinear_upsample_matrix(student_grid, target_grid)
            # Register as buffer (not a Parameter — bilinear weights are FIXED)
            self.register_buffer("W_up", W_up)
            self.perceiver_decoder = None
        elif pool_type == "perceiver_decoder":
            # v129 PCSD: Perceiver cross-attention decoder (NON-LINEAR, PUBLIC).
            # Replaces v128's linear bilinear interp with non-linear cross-attention
            # to break the rank-16 bottleneck. Applied CLIENT-SIDE on plaintext.
            # No W_up buffer (decoder handles spatial expansion via attention).
            #
            # v167: Plug-and-play perceiver. Two construction paths:
            #   1. perceiver_instance (BasePerceiverDecoder): use directly
            #      (bypasses registry + perceiver_config). Useful for externally
            #      constructed perceivers (e.g., test harness, custom subclass).
            #   2. perceiver_name (str): load from PerceiverRegistry. Defaults
            #      to "v162-default" (PerceiverDecoder, backward compat).
            self.W_up = None

            if perceiver_instance is not None:
                # v167 path 1: externally injected perceiver
                if not isinstance(perceiver_instance, BasePerceiverDecoder):
                    raise TypeError(
                        f"perceiver_instance must be BasePerceiverDecoder, got "
                        f"{type(perceiver_instance).__name__}"
                    )
                self.perceiver_decoder = perceiver_instance
            elif perceiver_name == "v162-default" and not perceiver_config:
                # v167 path 2a: default PerceiverDecoder (backward compat with
                # v129-v166 checkpoints). Construct directly to preserve exact
                # original __init__ behavior (avoids Registry's dim/num_latents
                # aliasing on PerceiverDecoder).
                pcfg = dict(
                    llm_dim=llm_dim,
                    num_queries=target_grid * target_grid,  # 256
                    num_input_tokens=num_tokens,  # 16
                    num_heads=4,
                    attention_dim=512,
                    ffn_dim=llm_dim,  # compact FFN (not 4× expansion)
                    num_layers=1,
                    init_from_bilinear=True,  # warm-start from v128 baseline
                    student_grid=student_grid,
                    target_grid=target_grid,
                )
                # v155 Innovation C+D: pass medical priors to PerceiverDecoder.
                if anatomical_prior is not None:
                    pcfg["anatomical_prior"] = anatomical_prior
                if disease_prior_embeddings is not None:
                    pcfg["disease_prior_embeddings"] = disease_prior_embeddings
                # v156 Innovation H: pass region_salience for per-region ICMC β
                if region_salience is not None:
                    pcfg["region_salience"] = region_salience
                    pcfg["num_regions"] = num_regions
                self.perceiver_decoder = PerceiverDecoder(**pcfg)
            else:
                # v167 path 2b: load arbitrary perceiver from registry
                # (e.g., "identity" for ablation, or future custom perceivers).
                # Pass all available kwargs; PerceiverRegistry.load forwards them.
                registry_kwargs = dict(
                    dim=llm_dim,
                    num_latents=target_grid * target_grid,  # 256
                    num_input_tokens=num_tokens,  # 16
                )
                # Forward medical priors if the perceiver supports them
                # (IdentityPerceiver ignores them; future variants may use them)
                if anatomical_prior is not None:
                    registry_kwargs["anatomical_prior"] = anatomical_prior
                if disease_prior_embeddings is not None:
                    registry_kwargs["disease_prior_embeddings"] = disease_prior_embeddings
                if region_salience is not None:
                    registry_kwargs["region_salience"] = region_salience
                    registry_kwargs["num_regions"] = num_regions
                # Perceiver-specific config overrides
                if perceiver_config:
                    registry_kwargs.update(perceiver_config)
                self.perceiver_decoder = PerceiverRegistry.load(
                    perceiver_name, **registry_kwargs,
                )
        else:
            self.W_up = None
            self.perceiver_decoder = None

        # Concept projection: D → LLM_dim (shared across all output tokens)
        # v143: LINEAR projection (SpectralBoundedLinear, τ=3.0).
        #   v142 polynomial kernel FAILED (distorts feature distribution → LLM
        #   regression). v143 reverts to v140a linear W_concept and breaks the
        #   rank-16 bottleneck via the PARAMETER-FREE image-guided decoder's
        #   GELU+softmax non-linearity (client-side, no FHE cost).
        #   FHE depth: 1 (cipher×plain matmul); total 7 < 13 budget.
        self.W_concept = SpectralBoundedLinear(
            dim, llm_dim, bias=True, spectral_bound=3.0,
        )
        # v156 Innovation G: MedWConcept — disease-anchored projection init.
        # Re-init W_concept to align the 64-dim FHE features to MedGemma's
        # disease-concept directions, counteracting the LLM's negative-prior
        # bias ("no findings" over-prediction). Init-only (ZERO FHE depth change).
        if disease_anchor_directions is not None:
            self.init_disease_anchored(disease_anchor_directions, disease_anchor_alpha)

        # Public 4×4 spatial canvas added after pooled FDBR decode. It does not
        # change the released coefficients; it only restores a server-side
        # positional template that global/anatomical pooling discarded.
        self._public_spatial_prior = None
        self.public_spatial_scale = 0.5

        # Learnable medical concept tokens (plaintext, client-side)
        # v124-v126 legacy: "soft prompts" that prime the LLM
        # v127+ SBA/CSB/PCSD/Direct: NOT used (256 tokens match SigLIP native count).
        # Allocate concept tokens only for pool types that consume them.
        if pool_type in ("global", "region"):
            self.concept_tokens = nn.Parameter(torch.zeros(1, num_concepts, llm_dim))
            nn.init.trunc_normal_(self.concept_tokens, std=0.02)
        else:
            self.concept_tokens = None

        # Cached cipher-path weights (v143: single W + bias, linear projection)
        self._W_concept_t: Optional[torch.Tensor] = None  # [D, LLM_dim] for TenSEAL
        self._b_concept: Optional[torch.Tensor] = None    # [LLM_dim]
        self.diagnostic_bottleneck: Optional[DiagnosticBottleneckRelease] = None
        self._num_tokens_inv: float = 1.0 / max(num_tokens, 1)

        # v125 region-aware pooling: token_regions[i] = region index of token i
        # Set by encoder via set_token_regions() (matches AAPN regions)
        self.token_regions: Optional[List[int]] = None
        # Pre-computed region->token indices for efficient cipher pooling
        self._region_token_indices: Optional[List[List[int]]] = None
        self._region_inv_sizes: Optional[List[float]] = None

    def set_token_regions(self, token_regions: List[int]):
        """Set token→region mapping (called by encoder when pool_type='region').

        Args:
            token_regions: List of N region indices (0..K-1). Must have
                len == num_tokens and max < num_regions.
        """
        if len(token_regions) != self.num_tokens:
            raise ValueError(
                f"token_regions length {len(token_regions)} != num_tokens {self.num_tokens}"
            )
        self.token_regions = list(token_regions)
        # Pre-compute region → token indices
        self._region_token_indices = [
            [i for i, r in enumerate(self.token_regions) if r == k]
            for k in range(self.num_regions)
        ]
        # Pre-compute 1/|region_k| for each region (handles empty regions)
        self._region_inv_sizes = [
            1.0 / max(len(indices), 1) for indices in self._region_token_indices
        ]

    def forward_plain(self, x: torch.Tensor) -> torch.Tensor:
        """Plain forward (training mode).

        x: [B, N, D] tensor (ViT features)
        Returns:
            - 'global'/'region': [B, K+num_concepts, LLM_dim] (K=1 or num_regions)
            - 'spatial_upsample'/'perceiver_decoder': [B, 256, LLM_dim]
              (no concept tokens concatenated)

        Architecture (global pool, v124):
          pooled = mean(x, dim=1)              [B, D]
          projected = pooled @ W_concept + b   [B, LLM_dim]
          projected = projected.unsqueeze(1)   [B, 1, LLM_dim]
          out = cat([projected, concept_tokens], dim=1)  [B, 1+R, LLM_dim]

        Architecture (region pool, v125):
          For each region k:
            pooled_k = mean(x[:, region_k_tokens, :], dim=1)  [B, D]
            projected_k = pooled_k @ W_concept + b            [B, LLM_dim]
          projected = stack(projected_k, dim=1)               [B, K, LLM_dim]
          out = cat([projected, concept_tokens], dim=1)       [B, K+R, LLM_dim]

        Architecture (spatial_upsample, v127 SBA / v128 CSB — FUSED):
          # Project first (16 matmuls), then bilinear interp (cheap)
          projected = x @ W_concept + b           [B, 16, LLM_dim]  (matmul)
          # Reshape to spatial grid, upsample, reshape back
          projected = projected.permute(0, 2, 1)  [B, LLM_dim, 16]
          projected = projected.reshape(B, LLM_dim, student_grid, student_grid)
          upsampled = F.interpolate(projected, size=(target_grid, target_grid),
                                     mode='bilinear', align_corners=False)
          out = upsampled.reshape(B, LLM_dim, 256).permute(0, 2, 1)  [B, 256, LLM_dim]
          # No concept tokens (256 matches SigLIP native count)

        Architecture (perceiver_decoder, v129 PCSD):
          # Step 1: Project 16 tokens to LLM_dim (matmul, 16×)
          projected = x @ W_concept + b           [B, 16, LLM_dim]
          # Step 2: Perceiver cross-attention (NON-LINEAR, breaks rank-16)
          out = perceiver_decoder(projected)      [B, 256, LLM_dim]
          # No concept tokens (256 matches SigLIP native count)
        """
        B, N, D = x.shape

        if self.pool_type == "direct":
            # v144: Direct projection — NO pooling, NO decoder, NO upsampling.
            # W_concept projects ALL 256 tokens directly: [B, 256, D] → [B, 256, llm_dim].
            # Student grid (16×16=256) == target grid (16×16=256, SigLIP native).
            # NO rank bottleneck (rank = 256). NO decoder → NO parameter sharing.
            out = self.W_concept(x)  # [B, 256, llm_dim]
            return out  # no concept tokens (256 matches SigLIP native count)

        if self.pool_type == "spatial_upsample":
            # v127 SBA: FUSED project-first-then-interp (algebraically equivalent
            # to interp-first-then-project, but 16× faster under FHE)
            # Step 1: Project 16 tokens to LLM_dim [B, 16, LLM_dim]
            # v143: linear projection (x@W + b) via W_concept(x)
            projected = self.W_concept(x)  # [B, 16, LLM_dim]
            # Step 2: Bilinear interp 4×4 → 16×16 = 256 tokens
            # Reshape: [B, 16, LLM_dim] → [B, LLM_dim, 4, 4]
            projected = projected.permute(0, 2, 1)  # [B, LLM_dim, 16]
            projected = projected.reshape(
                B, self.llm_dim, self.student_grid, self.student_grid
            )
            # F.interpolate with align_corners=False matches our W_up matrix
            upsampled = F.interpolate(
                projected,
                size=(self.target_grid, self.target_grid),
                mode='bilinear',
                align_corners=False,
            )
            # Reshape back: [B, LLM_dim, 16, 16] → [B, 256, LLM_dim]
            out = upsampled.reshape(B, self.llm_dim, -1).permute(0, 2, 1)
            return out  # [B, 256, LLM_dim] — no concept tokens

        elif self.pool_type == "perceiver_decoder":
            # v129 PCSD: Project + Perceiver cross-attention (NON-LINEAR).
            # Step 1: Project 16 tokens to LLM_dim [B, 16, LLM_dim] (SAME as v128)
            if self.diagnostic_bottleneck is not None:
                projected = self.diagnostic_bottleneck.projected_reference(x)
            else:
                projected = self.W_concept(x)  # [B, 16, LLM_dim]
            # NOTE: CFP (Calibrated Feature Perturbation) is NOT applied here.
            # Governed DP perturbation is a client-side operation:
            # the client adds noise AFTER decrypting the FHE output, BEFORE
            # sending the features to the server for the Perceiver. Injecting
            # noise in this server-side forward path would (1) break the
            # FHE/plain cosine verification and (2) violate the client-side
            # noise contract. forward_plain must remain deterministic.
            # Step 2: Perceiver decoder (NON-LINEAR cross-attention)
            # Breaks v128's rank-16 bottleneck: output rank can reach 256.
            out = self.perceiver_decoder(projected)  # [B, 256, LLM_dim]
            return out  # [B, 256, LLM_dim] — no concept tokens

        elif self.pool_type == "global" or self.token_regions is None:
            # v124 global pool
            pooled = x.mean(dim=1)  # [B, D]
            # v143: linear projection (pooled@W + b)
            projected = self.W_concept(pooled)  # [B, LLM_dim]
            projected = projected.unsqueeze(1)  # [B, 1, LLM_dim]
        else:
            # v125 region-aware pool
            projected_list = []
            for k, indices in enumerate(self._region_token_indices):
                if len(indices) == 0:
                    # Empty region: zero token (W_concept of a zero input = bias only)
                    zero_input = torch.zeros(B, self.dim, device=x.device, dtype=x.dtype)
                    projected_k = self.W_concept(zero_input)  # [B, LLM_dim]
                    projected_list.append(projected_k)
                else:
                    pooled_k = x[:, indices, :].mean(dim=1)  # [B, D]
                    # v143: linear projection
                    projected_k = self.W_concept(pooled_k)  # [B, LLM_dim]
                    projected_list.append(projected_k)
            projected = torch.stack(projected_list, dim=1)  # [B, K, LLM_dim]

        # Concatenate with concept tokens (broadcast to batch)
        concepts = self.concept_tokens.expand(B, -1, -1)  # [B, R, LLM_dim]
        out = torch.cat([projected, concepts], dim=1)  # [B, K+R, LLM_dim]
        return out

    def forward_cipher(self, x: CiphertextTensor) -> CiphertextTensor:
        """Cipher forward (FHE inference mode).

        x: CiphertextTensor with N ciphers, each D-slot
        Returns: CiphertextTensor with K ciphers, each LLM_dim-slot
            K=1 for global pool, K=num_regions for region pool,
            K=16 for spatial_upsample (v128 CSB — server projects 16 ciphers,
                client does bilinear interp on PLAINTEXT after decryption to
                produce 256 tokens; DEPRECATED, v129 PCSD
                is the canonical replacement).

        FHE implementation (global pool, v124):
          pooled = sum(ciphers) * (1/N)        (+1 depth: cipher×scalar)
          projected = pooled.matmul(W_concept) (+1 depth: matmul_plain)
          projected = projected + b_concept    (0 depth: cipher+plain)
          Total depth: 2

        FHE implementation (region pool, v125):
          For each region k:
            pooled_k = sum(ciphers[i] for i in region_k) * (1/|region_k|)
                                                     (+1 depth: cipher×scalar)
            projected_k = pooled_k.matmul(W_concept)  (+1 depth: matmul_plain)
            projected_k = projected_k + b_concept     (0 depth: cipher+plain)
          Total depth: 2 (SAME as global — regions are parallel, not sequential)

        FHE implementation (spatial_upsample, v128 CSB — CLIENT-SIDE INTERP):
          v128 INNOVATION: Privacy-Computation Splitting.
          The bilinear interp weights W_up are PUBLIC (derived from grid sizes),
          so doing them under FHE wastes cipher ops with NO privacy benefit.
          v128 moves the bilinear interp to the CLIENT (plaintext domain).

          SERVER (FHE, this method):
            For each of 16 input ciphers:
              projected[i] = ciphers[i].matmul(W_concept)  (+1 depth)
              projected[i] = projected[i] + b_concept      (0 depth)
            Return 16 ciphers (NOT 256!).

          CLIENT (plaintext, after decryption — v128 bilinear interp, DEPRECATED
            in favor of v129 PCSD's Perceiver decode):
            plain_16 = decrypt(projected)        # [16, llm_dim]
            plain_256 = W_up @ plain_16          # [256, 16] @ [16, llm_dim]
            Feed plain_256 to LLM (matches SigLIP's native 256-token count).

          Total FHE depth: 1 (matmul ONLY — depth REDUCTION from v127 SBA's 2)
          Total cipher ops: 16 matmuls (vs v127 SBA's 16 matmuls + ~1800 ops)
          Algebraically equivalent to v127 SBA (proven in module docstring).

        NOTE: For 'global'/'region' pool types, concept tokens are NOT added
        in cipher mode — they are plaintext and added by the client AFTER
        decryption. For 'spatial_upsample' (v128), no concept tokens are used
        (256 tokens match SigLIP native count, no padding needed).
        """
        engine = x._engine
        ciphers = x._cipher  # list of N TenSEAL CKKSVectors, each D-slot

        if len(ciphers) == 0:
            raise ValueError("Cannot pool empty cipher list")

        if self.diagnostic_bottleneck is not None:
            return self.diagnostic_bottleneck.encode_cipher(x)

        if self.pool_type in ("spatial_upsample", "perceiver_decoder", "direct"):
            # v128 CSB / v129 PCSD / v144 Direct — SERVER-SIDE PROJECTION.
            # The spatial upsampling (v128 bilinear interp), non-linear decoding
            # (v129 Perceiver), or direct projection (v144, NO transform) all use
            # PUBLIC or ZERO weights (or NO transform at all), so any client-side
            # step is trivial or absent.
            #
            # LINEAR projection (x@W + b) per cipher.
            #   Cipher ops per token: 1 matmul (cipher×plain)
            #   FHE depth: 1 (was 2 in v142 polynomial — simpler, less noise)
            #   Privacy: W_concept is PRIVATE server weight under FHE.
            # v144: projects ALL 256 ciphers (vs v128/v129's 16). Client decrypts 256
            #   and feeds directly to LLM — NO decoder, NO transform (identity).
            projected_ciphers = [
                self._linear_project_one(ct) for ct in ciphers[:self.num_tokens]
            ]  # N ciphers (16 for v128/v129, 256 for v144), each at depth+1
            # Return N ciphers — client will decrypt and apply:
            #   v128: bilinear interp (linear) → 256 tokens (rank ≤ 16)
            #   v129: Perceiver decoder (non-linear) → 256 tokens (rank ≤ 256)
            #   v144: IDENTITY (no transform) → 256 tokens (rank = 256, NO decoder)
            out_depth = x.depth + 1  # linear matmul only (+1)
            return CiphertextTensor(
                ciphertexts=projected_ciphers,
                engine=engine,
                depth=out_depth,
            )

        elif self.pool_type == "global" or self.token_regions is None:
            # v124 global pool: 1 output cipher
            pooled = ciphers[0]
            for ct in ciphers[1:]:
                pooled = pooled + ct  # 0 depth
            pooled = pooled * self._num_tokens_inv  # +1 depth
            # v143: linear projection (depth +1 on top of pool +1 → +2 total)
            projected = self._linear_project_one(pooled)
            out_ciphers = [projected]
            out_depth = x.depth + 2  # pool(+1) + linear(+1)
        else:
            # v125 region-aware pool: K output ciphers
            out_ciphers = []
            for k, indices in enumerate(self._region_token_indices):
                if len(indices) == 0:
                    # Empty region: encrypt a zero vector + bias (linear of zero = bias)
                    b_list = self._b_concept.detach().cpu().reshape(-1).tolist()
                    zero_ct = engine.ts.ckks_vector(engine.context, [0.0] * self.llm_dim)
                    projected_k = zero_ct + b_list  # 0 depth (plain add)
                    out_ciphers.append(projected_k)
                    continue
                # Sum ciphers in region k (cipher+cipher, 0 depth)
                pooled_k = ciphers[indices[0]]
                for idx in indices[1:]:
                    pooled_k = pooled_k + ciphers[idx]  # 0 depth
                # Scale by 1/|region_k| (cipher×scalar, +1 depth)
                pooled_k = pooled_k * self._region_inv_sizes[k]  # +1 depth
                # v143: linear projection (depth +1 on top of pool +1 → +2 total)
                projected_k = self._linear_project_one(pooled_k)
                out_ciphers.append(projected_k)
            out_depth = x.depth + 2  # pool(+1) + linear(+1)

        return CiphertextTensor(
            ciphertexts=out_ciphers,
            engine=engine,
            depth=out_depth,
        )

    def _linear_project_one(self, ct):
        """v143: Linear projection on ONE raw TenSEAL CKKSVector.

        Computes: ct @ W_concept + bias  (standard linear matmul).

        Depth budget per call (relative to input ct's level):
          - proj = ct.matmul(W_concept)  [+1, cipher×plain matmul]
          - proj = proj + bias           [0, cipher+plain]
          → output level = input level - 1  (i.e. depth +1)

        Args:
            ct: raw TenSEAL CKKSVector (D slots, one token's features).
        Returns:
            raw TenSEAL CKKSVector (LLM_dim slots, linearly projected).
        """
        proj = ct.matmul(self._W_concept_t)  # +1 depth (cipher matmul plain)
        b_list = self._b_concept.detach().cpu().reshape(-1).tolist()
        proj = proj + b_list                  # 0 depth (cipher + plain)
        return proj

    def prepare_cipher_weights(self):
        """Cache weights for cipher path (v143: single W + bias, linear projection)."""
        # TenSEAL matmul expects [in, out] = [D, LLM_dim].
        # SpectralBoundedLinear.weight is ALREADY [in, out] = [D, LLM_dim].
        # CRITICAL: TenSEAL can't operate on torch.Parameter (requires_grad=True).
        # Must detach() and convert to plain tensor.
        self._W_concept_t = self.W_concept.weight.detach().cpu()  # [D, LLM_dim]
        self._b_concept = (self.W_concept.bias.detach().cpu()
                           if self.W_concept.bias is not None
                           else torch.zeros(self.llm_dim))

    def fuse_input_affine(self, gamma: torch.Tensor, beta: torch.Tensor) -> None:
        """Fold a preceding affine ``gamma*x + beta`` into W_concept.

        This is exact for the linear release projection and removes one CKKS
        level in callback-free mode.  The operation is idempotent per adapter
        instance and intentionally mutates only the deployment copy.
        """
        if getattr(self, "_input_affine_fused", False):
            return
        gamma = gamma.detach().cpu().reshape(-1)
        beta = beta.detach().cpu().reshape(-1)
        if gamma.numel() != self.dim or beta.numel() != self.dim:
            raise ValueError("input affine dimensions must match adapter input dim")
        with torch.no_grad():
            weight = self.W_concept.weight.detach().cpu()
            bias = (
                self.W_concept.bias.detach().cpu()
                if self.W_concept.bias is not None
                else torch.zeros(self.llm_dim)
            )
            self.W_concept.weight.copy_(gamma.unsqueeze(1) * weight)
            self.W_concept.bias.copy_(beta @ weight + bias)
        self._input_affine_fused = True
        self._W_concept_t = None
        self._b_concept = None

    def configure_diagnostic_bottleneck(
        self,
        basis: torch.Tensor,
        token_groups: Optional[List[List[int]]] = None,
    ) -> None:
        """Configure FDBR after checkpoint loading and before cipher baking."""
        self.diagnostic_bottleneck = DiagnosticBottleneckRelease(
            basis=basis,
            output_weight=self.W_concept.weight,
            output_bias=self.W_concept.bias,
            token_groups=token_groups,
        )

    def configure_public_spatial_prior(
        self,
        pos_embed: torch.Tensor,
        scale: float = 0.5,
    ) -> None:
        """Build a public residual spatial template for pooled FDBR tokens.

        When FDBR mean-pools tokens, decode broadcasts one vector to every
        member of a group. The Perceiver then attends to duplicated keys.
        `pos_embed` and `W_concept` are server-owned, so adding the diagnostic-
        orthogonal positional residual does not enlarge the client release.
        """
        if self.diagnostic_bottleneck is None:
            raise RuntimeError("configure the diagnostic bottleneck before the spatial prior")
        if scale < 0:
            raise ValueError("public spatial scale must be non-negative")
        groups = self.diagnostic_bottleneck.token_groups
        if groups is None or all(len(group) <= 1 for group in groups):
            self._public_spatial_prior = None
            self.public_spatial_scale = float(scale)
            return
        pos = torch.as_tensor(pos_embed).detach().float()
        if pos.ndim == 3:
            pos = pos.reshape(pos.shape[1], pos.shape[2])
        if pos.ndim != 2:
            raise ValueError("pos_embed must have shape [T, D] or [1, T, D]")
        token_count = self.diagnostic_bottleneck.input_token_count
        if pos.shape[0] < token_count:
            raise ValueError(
                f"pos_embed has {pos.shape[0]} tokens, bottleneck expects {token_count}"
            )
        pos = pos[:token_count]
        if pos.shape[1] != self.dim:
            raise ValueError(
                f"pos_embed dim {pos.shape[1]} != encoder dim {self.dim}"
            )
        basis = self.diagnostic_bottleneck.basis.to(dtype=pos.dtype)
        residual = pos - pos @ basis @ basis.T
        # SpectralBoundedLinear stores [in, out] = [D, LLM_dim].
        weight = self.W_concept.weight.detach().float()
        prior = residual @ weight
        self._public_spatial_prior = prior.contiguous()
        self.public_spatial_scale = float(scale)

    def _add_public_spatial_prior(self, decoded: torch.Tensor) -> torch.Tensor:
        if self._public_spatial_prior is None or self.public_spatial_scale == 0:
            return decoded
        prior = self._public_spatial_prior.to(device=decoded.device, dtype=decoded.dtype)
        if decoded.shape[-2] != prior.shape[0] or decoded.shape[-1] != prior.shape[1]:
            raise ValueError(
                f"spatial prior {tuple(prior.shape)} does not match decoded {tuple(decoded.shape)}"
            )
        centered = prior - prior.mean(dim=0, keepdim=True)
        token_rms = decoded.detach().pow(2).mean().sqrt().clamp_min(1e-6)
        prior_rms = centered.pow(2).mean().sqrt().clamp_min(1e-6)
        return decoded + (self.public_spatial_scale * token_rms / prior_rms) * centered

    def server_side_bottleneck_decode(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Decode released diagnostic coefficients using private server weights."""
        if self.diagnostic_bottleneck is None:
            raise RuntimeError("Diagnostic bottleneck is not configured")
        return self._add_public_spatial_prior(
            self.diagnostic_bottleneck.decode_plain(coefficients)
        )

    def apply_dsr(self, dsr_projection: torch.Tensor, space: str = "input"):
        """v176: Fold DSR (Diagnostic Subspace Retention) into W_concept.

        STRONG THREAT MODEL DEFENSE: Removes patient-identity directions from
        features via deterministic subspace projection. Zero additional FHE
        depth (folded into existing W_concept matmul), zero client compute.

        MUST be called AFTER load_state_dict() (otherwise weight loading
        overwrites the folded W_concept). The pipeline calls this in
        from_pretrained() after loading encoder weights.

        Args:
            dsr_projection: [d, d] symmetric idempotent projector Π = U @ U^T.
                For space="input": d = D_in (ViT dim, e.g. 64) — UNIVERSAL,
                    shared across all LLMs. Calibrated on ViT features.
                For space="output": d = D_out (LLM_dim, e.g. 2560) — LLM-specific.
            space: "input" (default, universal) or "output" (LLM-specific).
        """
        with torch.no_grad():
            proj = dsr_projection.to(
                self.W_concept.weight.device
            ).to(self.W_concept.weight.dtype)
            if space == "input":
                # Π @ W: [D_in, D_in] @ [D_in, D_out] = [D_in, D_out]
                # Removes patient dirs from ViT features BEFORE projection.
                self.W_concept.weight.data = proj @ self.W_concept.weight.data
            elif space == "output":
                # W @ Π: [D_in, D_out] @ [D_out, D_out] = [D_in, D_out]
                # Removes patient dirs from LLM_dim features AFTER projection.
                self.W_concept.weight.data = self.W_concept.weight.data @ proj
            else:
                raise ValueError(f"space must be 'input' or 'output', got {space!r}")
            # Clear cached cipher weights so they get re-baked with the new W_concept
            self._W_concept_t = None
            self._b_concept = None

    def init_disease_anchored(
        self,
        disease_anchor_directions: torch.Tensor,
        alpha: float = 0.3,
    ):
        """v156 Innovation G: Re-initialize W_concept with disease-anchor alignment.

        Blends Xavier init with disease-anchor directions extracted from MedGemma
        (SVD of image-token embeddings projected onto disease-term directions).
        This aligns the 64-dim FHE features to the LLM's disease-concept
        directions, counteracting the LLM's negative-prior bias ("no findings"
        over-prediction due to training data imbalance).

        FIRST-PRINCIPLES ANALYSIS:
          Medical LLMs (MedGemma) trained on imbalanced radiology data over-
          predict "no findings" — the text prior dominates over weak visual
          signal. A plain Xavier W_concept (64→2560) does NOT align FHE
          features to the LLM's disease-concept directions, so the LLM's text
          prior dominates. By initializing W_concept to project PREFERENTIALLY
          onto disease-anchor directions, the LLM gets a medical-meaningful
          starting point, counteracting negative-prior bias.

        FHE IMPACT: STILL linear matmul, +1 depth. ZERO noise increase
          (init-only — no architectural change, just better starting weights).

        Args:
            disease_anchor_directions: [D, llm_dim] = [64, 2560] tensor. Each
                column is a disease-anchor direction in LLM space, projected to
                the 64-dim input subspace (computed by calibration via SVD of
                MedGemma image-token embeddings onto disease-term directions).
            alpha: Blend factor (default 0.3). W_concept = α·D_proj + (1−α)·W_xavier.
                At α=0: pure Xavier (v154 baseline). At α=1: pure disease-anchored.
                α=0.3 is a conservative blend — disease directions guide but don't
                dominate, leaving room for training to refine.
        """
        assert disease_anchor_directions.shape == (self.dim, self.llm_dim), (
            f"disease_anchor_directions shape {disease_anchor_directions.shape} != "
            f"({self.dim}, {self.llm_dim})"
        )
        with torch.no_grad():
            # Current Xavier init [D, llm_dim] (SpectralBoundedLinear.weight is [in, out])
            W_xavier = self.W_concept.weight.clone()
            D_proj = disease_anchor_directions.to(W_xavier.dtype)
            # Blend: disease-anchored directions + Xavier diversity
            new_W = alpha * D_proj + (1.0 - alpha) * W_xavier
            self.W_concept.weight.copy_(new_W)
        # Re-project to spectral bound τ=3.0 (disease directions may exceed bound)
        self.W_concept.project_spectral()

    def multiplicative_depth(self) -> int:
        """Return FHE depth of adapter.

        v143 (linear W_concept: x@W + b, depth 1):
        - 'global'/'region' (v124/v125/v126): 2 (pool +1, linear +1)
        - 'spatial_upsample' (v128 CSB): 1 (linear +1 ONLY — bilinear interp
          is done CLIENT-SIDE on plaintext, +0 FHE depth)
        - 'perceiver_decoder' (v129 PCSD): 1 (linear +1 ONLY — Perceiver decoder
          is done CLIENT-SIDE on plaintext, +0 FHE depth)
        - 'direct' (v144): 1 (linear +1 ONLY — NO decoder, NO transform, client
          just decrypts 256 tokens and feeds to LLM directly)

        Total pipeline depth: encoder(6) + adapter(1) = 7 < 13 budget (6 margin).
        v142 had depth 8 (polynomial +2); v143/v144's depth 7 is simpler and lower-noise.
        """
        if self.pool_type in ("spatial_upsample", "perceiver_decoder", "direct"):
            return 1  # v143/v144: linear projection depth 1
        return 2  # pool(+1) + linear(+1)

    def server_side_perceiver_decode(self, plain_tokens: torch.Tensor) -> torch.Tensor:
        """Apply Perceiver decoder on plaintext tokens (SERVER-side, v149+).

        This method runs on the SERVER with PRIVATE Perceiver weights
        (satisfies the "server does not share model params" constraint). The
        server holds the private Perceiver weights and runs this method after
        receiving the (CFP-noised) plaintext features from the client.

        v154 CFP: The CLIENT applies CFP noise (via
        ``fpavlm.cipher_native.dp_accountant.DPCFPAccountant``) to the
        features AFTER decryption and BEFORE sending them to the server. The
        server then calls this method on the noised features. CFP is NOT
        applied inside this method or in ``forward_plain`` — it is a purely
        client-side operation.

        This is the SERVER-side half of v129 PCSD. After the client decrypts
        the 16 projected ciphers (and applies CFP noise), the server runs this
        method to produce 256 spatially diverse tokens matching SigLIP's
        native token count.

        DIFFERENCE from v128 CSB (linear bilinear interp, DEPRECATED — v129 PCSD is canonical):
            v128: linear bilinear interp, output rank ≤ 16 (RANK BOTTLENECK)
            v129: non-linear cross-attention, output rank ≤ 256 (FULL RANK)
            The softmax in attention breaks the rank-16 bottleneck of v128.
        """
        if self.pool_type != "perceiver_decoder" or self.perceiver_decoder is None:
            raise RuntimeError(
                f"server_side_perceiver_decode requires pool_type='perceiver_decoder', "
                f"got {self.pool_type!r}"
            )
        # v154 rename: perceiver_decoder.client_side_decode → _server_side_decode_internal
        # (the Perceiver is SERVER-side with PRIVATE weights in v149+, so the old
        # "client_side_decode" name was misleading). Body unchanged.
        return self.perceiver_decoder._server_side_decode_internal(plain_tokens)

    def bake_statistics(self):
        """Bake statistics for cipher path."""
        self.prepare_cipher_weights()
