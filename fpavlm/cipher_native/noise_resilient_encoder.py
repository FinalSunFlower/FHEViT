"""Noise-resilient medical ViT encoder for CKKS inference.

v165 = v164 + CADN (Client-Assisted Dynamic Normalization, CA-FHE paradigm instance 3).
v164 = v153 (CAS + High-Precision CKKS) + v154 CFP + v156 medical specializations
(A–H) + v164 targeted loss stack (FN-Distill + FN-RKD + RLM-PAM + FNC-ICMC + FN-XDiv).
The encoder design inherits the v124 LPPN block and adds v152 CAS, v153 high-precision
CKKS (N=32768), v154 CFP (client-side deterministic privacy), v156 medical priors,
v164 amplification scales (SYM_SCALE=10, BETA_EFFECT_SCALE=5→20 in v165fix), and v165 CADN (exact
LayerNorm via client plaintext computation, depth-0). See
`docs/handoff/ARCHITECTURE_CURRENT.md` for the current scope and trust boundary.
  - SERVER: ALL private-weight computation under FHE (CADN/PolyNorm + AAPN + PolyAct LPPN + PolyMLP)
  - CLIENT: CANR refresh, optional CAS/CADN, final release decryption, and DP noise.
  - v165 historical CKKS smoke test (5 OpenI images, beam search num_beams=4 + seed=42):
    cos_16=1.000000 on the recorded five-image run,
    cos_256=1.000009, rel_l2_16=0.000654 (25× better than v164's 0.0167),
    TF-IDF(FHE,GT)=0.0903 (≥0.09 PASS, +17.1% over v164 greedy's 0.0771)
  - v124 LPPN: [1,1,1] uniform linear PolyAct (eliminates ALL cipher×cipher in
    activation). cos +0.000258 vs v122 [2,1,1] baseline (258× noise floor).
  - v165 CADN: replaces PolyNorm (depth-2 polynomial approximation) with EXACT client-
    computed LayerNorm (depth-0, dynamic, without polynomial approximation). Zero architectural train-inference
    gap (same LayerNorm function in train+inference; only float-precision + CKKS noise
    differ, ~1e-9, negligible vs PolyNorm's ~1e-2 polynomial error).

  ⚠ HISTORICAL CAVEAT (v124 era, RESOLVED in v162+):
    v124's TF-IDF(FHE,Ref)=1.0 was semantically VACUOUS (all 5 FHE reports IDENTICAL
    regardless of pathology). v162+ resolved this via RKD + Margin Contrastive + ICMC
    Prior + unfrozen encoder. v165 achieves 5/5 distinct reports, TF-IDF(FHE,GT)=0.0903
    (≥0.09 PASS, via beam search num_beams=4 + seed=42). The data-scale weakness was
    improved in v165fix-beam, but the small evaluation does not establish SOTA.

The encoder orchestrates transformer blocks under FHE. In cipher mode, each
block requires a `client_callback` that performs CANR refresh (decrypt →
re-encrypt), resetting multiplicative depth to 0.

PRIVACY MODEL (v116+ FHE-first):
  - Server (encoder) receives a `client_callback`, NOT the client engine
  - Callback signature: fn(ct_list: list) -> list (fresh cts, pure CANR)
  - Server CANNOT call decrypt() — only the callback is exposed
  - In deployment: callback is an RPC stub; in verification: wraps ClientCANREngine
  - Server NEVER sees: raw image, plaintext features, secret_key
  - Client NEVER sees: model weights — only holds secret_key for CANR

CKKS PARAMETERS (v165 high-precision + CADN):
  - N=32768 (full, REQUIRED for cos=1.000000), scale=2^28, depth_budget=13
  - coeff_mod = [60, 28×13, 60] = 484 bits < 881-bit 128-sec limit (N=32768, classical)
  - slot_count=16384 (fits dim=64, mlp_hidden=128 with room)
  - v165 CADN per-block depth: 4 (with LTM) < 13 budget (9 margin)
  - v164 (non-CADN) max segment depth: 6 (Block + final_norm + W_concept) < 13 budget

SUPPORTED BLOCK TYPES (3 total — v110/v111/v114/v115 DELETED in v125 cleanup):
  - 'med_fhe_vit_v3' (v118+, retained historical baseline): PolyNorm + AAPN + PolyAct
  - 'med_fhe_vit_v2' (v117): PolyNorm + AAPN + Hadamard activation
  - 'med_fhe_vit'   (v116): PolyNorm + shared coeffs + Hadamard

DELETED IN v125 CLEANUP (2026-07-21):
  - 'hybrid_attention' (v110): Split Learning, VIOLATES FHE-first
  - 'fhe_native' (v111): FAILED v111.3, cos=0.862
  - 'med_mixer' (v114): FHE-native dense token-mixing, slow at scale
  - 'med_loc_mixer' (v115): FHE-native dilated local token-mixing
  See `docs/handoff/STATUS.md` and `docs/handoff/CLAIMS_AUDIT.md` for the current
  evidence and claim boundaries.
"""
from __future__ import annotations

import gc
import math
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from .adaptive_noise import AdaptiveCKKSNoise
from .anatomy_router import AnatomyAwareTokenRouter
from .cadn_norm import CADNLayer
from .cipher_norm import CipherRMSNorm
from .ciphertext_tensor import CiphertextTensor
from .med_fhe_vit_block import MedFHEBlock
from .medical_concept_adapter import MedicalConceptAdapter
from .refresh_controller import (
    RefreshAction,
    RefreshController,
    merge_ciphertexts,
    select_ciphertexts,
)
from .spectral_bounded import SpectralBoundedLinear


class NoiseResilientMAViTEncoder(nn.Module):
    """FHE-ViT Encoder: orchestrates v116-v124 MedFHEBlocks under FHE.

    v124 LPPN Architecture (block_type='med_fhe_vit_v3', historical baseline):
      Per block (degree=1 LPPN, depth 5):
        [SERVER] PolyNorm:    z = c0 + c1·x + c2·x²           → depth 2
        [SERVER] PolyAct W_a: h = W_a · z                       → depth 3
        [SERVER] PolyAct σ:   σ = a0 + a1·h (cipher×plain)      → depth 4
        [SERVER] PolyMLP W_o: out = W_o · σ                     → depth 5
        [SERVER] Residual:    x_next = x + α·out                → depth 5
        [CANR]   → depth 0 (block-end, 1 CANR per block)

      v124 LPPN [1,1,1] uniform: ALL blocks degree=1 (NO cipher×cipher in PolyAct).
      Nonlinearity comes from PolyNorm's c2·x² (kept, 1 cipher×cipher per block).

      Client compute: 1 CANR/block × L blocks + 1 patch_proj CANR + 1 final CANR
      = L+2 CANR total. For L=3: 5 CANR (verified in v124 multi-image, N=5).

      Max block depth 5 < 13 budget (8 margin).
      See `docs/handoff/ARCHITECTURE_CURRENT.md` for the measured CKKS boundary.
    """

    def __init__(
        self,
        dim: int = 1152,
        num_layers: int = 12,
        mlp_ratio: float = 3.7362,
        num_patches: int = 1024,
        in_channels: int = 3,
        patch_size: int = 14,
        img_size: int = 448,
        spectral_bound: float = 1000.0,  # large = no projection (use SigLIP original)
        input_range: float = 3.0,
        # ACNT parameters (for optional plaintext training)
        noise_base_rate: float = 1e-3,
        noise_depth_growth: float = 0.05,
        noise_warmup_epochs: int = 5,
        enable_noise_training: bool = True,
        # Block type (only v116-v124 supported after v125 cleanup)
        block_type: str = "med_fhe_vit_v3",
        # Dropout for regularization (0 in FHE inference)
        dropout: float = 0.0,
        # v110: FFN spectral bound (large = no projection, use SigLIP original)
        ffn_spectral_bound: float = 1000.0,
        # v118 END-TO-END FHE PIPELINE: Optional MedicalConceptAdapter
        # When llm_dim > 0, adapter projects FHE features → LLM dim under FHE.
        # Client decrypts only 1 token × llm_dim (NOT the full N × D features).
        llm_dim: Optional[int] = None,
        num_concepts: Optional[int] = None,
        # v119/v124: PolyAct polynomial degree (1 or 2). Only used by med_fhe_vit_v3.
        #   degree=2 (v118): σ(h)=a0+a1·h+a2·h² (Horner: 1 cipher×plain +
        #     1 cipher×cipher). Full nonlinearity, but cipher×cipher creates
        #     CORRELATED noise that drops cos to ~0.9988 (FAIL < 0.999).
        #   degree=1 (v119/v124 historical setting): σ(h)=a0+a1·h (cipher×plain ONLY). Eliminates
        #     the dominant cipher×cipher noise. Linear activation — nonlinearity
        #     comes from PolyNorm's c2·x² term. Block depth 6 → 5.
        # v124 LPPN (Linear PolyAct + Polynomial Norm): [1,1,1] uniform schedule
        #   recorded small smoke-test value was cos=0.999711±0.000025 (N=5 images).
        #   Design philosophy: "Nonlinearity from Norm, not from Activation".
        poly_act_degree: int = 2,
        # v120/v124: Depth-adaptive degree schedule. When provided, overrides poly_act_degree
        # per-block. v124 LPPN discovery: [1,1,1] uniform BEATS [2,1,1] depth-adaptive
        # by +0.000258 cos (258× noise floor). The v120 depth-adaptive schedule is
        # SUPERSEDED by v124 LPPN for L≤3. For L>3, depth-adaptive may still help.
        # Examples:
        #   None (default): use uniform poly_act_degree for all blocks
        #   [2, 1, 1, 1]: block 0 uses degree=2, rest use degree=1 (good for L=4)
        #   "auto": automatically choose based on num_layers (v124 LPPN for L≤3)
        #     L≤3: all degree=1 (v124 LPPN historical setting, cos=0.999711±0.000025, N=5)
        #     L=4-6: first block degree=2, rest degree=1 (v120 depth-adaptive)
        #     L>6: all degree=1 (noise budget tight)
        poly_act_degree_schedule: Optional[List[int]] = None,
        # v121: num_regions parameter for per-region ablation (R ∈ {1,3,5,7,9}).
        # Only used by med_fhe_vit_v2/v3. Default 5 (chest X-ray anatomical regions).
        # R=1: shared PolyNorm coeffs (v116, no AAPN)
        # R>1: per-region PolyNorm coeffs (v117 AAPN)
        num_regions: int = 5,
        # v125/v128: MedicalConceptAdapter pooling type.
        #   'global' (v124): pool N tokens → 1 token (depth 2). Loses spatial info.
        #   'region' (v125): pool N tokens → K=num_regions tokens (depth 2, SAME).
        #     Preserves spatial pathology info (K=5 anatomical regions).
        #   'spatial_upsample' (v128 CSB): bilinear upsample 4×4 → 16×16 = 256
        #     tokens. SERVER projects 16 ciphers (depth 1), CLIENT does bilinear
        #     interp on plaintext (depth 0). Produces 256 spatially diverse tokens
        #     matching SigLIP's native token count for MedGemma's LLM.
        # v125 FIRST-PRINCIPLES: A chest X-ray report requires spatial localization
        #   ("left lower lobe consolidation", "cardiomegaly"). A single globally-
        #   pooled token cannot encode WHERE the pathology is. v125 region pool
        #   produces K tokens (one per anatomical region), preserving spatial info.
        # v128 FIRST-PRINCIPLES: MedGemma's LLM expects 256 image tokens in a 16×16
        #   grid (SigLIP native after avg_pool kernel=4). v124 (1 token) and v125/v126
        #   (5 region tokens, even repeated to 256) produce OOD-length inputs that
        #   cause the LLM to fall back to text-only prior ("no findings"). v128 CSB
        #   bilinearly upsamples 4×4 → 16×16 = 256 tokens, each with UNIQUE spatial
        #   position, matching SigLIP's native token count and spatial layout.
        # v128 PRIVACY-COMPUTATION SPLITTING (core innovation):
        #   The bilinear interp weights W_up are PUBLIC (derived from grid sizes
        #   4 and 16). Doing them under FHE wastes cipher ops with NO privacy
        #   benefit. v128 CSB moves bilinear interp to CLIENT-side plaintext:
        #     SERVER (FHE): 16 ciphers → 16 projected ciphers (16 matmuls, depth 1)
        #     CLIENT (plain): decrypt 16 ciphers → apply W_up → 256 tokens → LLM
        #   This is 100× faster than v127 SBA (which did interp under FHE) with
        #   IDENTICAL accuracy (bilinear is linear, decryption is linear).
        # FHE depth:
        #   global/region: pool (+1) + matmul (+1) = 2
        #   spatial_upsample (v128 CSB): matmul (+1) ONLY = 1  ← DEPTH REDUCTION
        pool_type: str = "global",
        # v128 CSB: Student/target grid sizes for spatial_upsample.
        # student_grid: Student FHE image grid (default 4 → 16 tokens)
        # target_grid: Target LLM image grid (default 16 → 256 tokens, matches SigLIP)
        student_grid: int = 4,
        target_grid: int = 16,
        # v129 PCSD: Perceiver decoder config (only used for pool_type='perceiver_decoder').
        # Keys: num_heads, attention_dim, ffn_dim, num_layers, init_from_bilinear.
        perceiver_config: Optional[dict] = None,
        # v167: Plug-and-play perceiver registry name. Loaded via
        # PerceiverRegistry in MedicalConceptAdapter. Default "v162-default"
        # uses PerceiverDecoder (backward compat). "identity" uses
        # IdentityPerceiver (ablation: skip perceiver, pass-through).
        perceiver_name: str = "v162-default",
        # v135: Linear Token Mixing (LTM) — FHE-native cross-token information flow.
        # When True, adds a LinearTokenMixer to each block (depth +1, cipher×plain only).
        # This enables spatial reasoning (cross-patch pathology detection) and breaks
        # the val_cos=0.82 ceiling. Default False (backward compat with v116-v134).
        enable_token_mix: bool = False,
        # v135: Spectral bound for token mixing matrix (FHE noise control)
        token_mix_spectral_bound: float = 3.0,
        # Callback-free optimization: fold affine degree-1 PolyAct into W_o.
        # Opt-in to preserve historical operation traces and checkpoints.
        fuse_linear_poly_act: bool = False,
        fuse_linear_poly_norm: bool = False,
        channelwise_polynorm: bool = False,
        fuse_final_affine: bool = False,
        # === v156 medical priors (all optional — None = v154 backward compat) ===
        # A: MedPolyNorm — dict with key "region_stats": [R, 2] (per-region mean, std).
        #    Initializes PolyNorm c0=-μ/σ, c1=1/σ, c2=α/σ (lesion-sensitive).
        medical_prior: Optional[dict] = None,
        # C: Anatomical Perceiver Queries — [256, llm_dim] anatomical prototypes.
        #    Initializes Perceiver queries with spatial-anatomical structure + symmetry.
        anatomical_prior: Optional[torch.Tensor] = None,
        # D: Disease-Prior Concept Injection — [K, llm_dim] disease-term embeddings.
        #    K=20 disease tokens modulate Perceiver output via cross-attention (SkipInit).
        disease_prior_embeddings: Optional[torch.Tensor] = None,
        # E: MedPolyAct — dict with key "region_salience": [R] per-region salience.
        #    Per-region PolyAct coeffs (a0_r, a1_r) tuned to radiological region salience.
        med_poly_act_prior: Optional[dict] = None,
        # F: MedLTM — List of (r_a, r_b) contralateral region pairs.
        #    Enables symmetry-difference token mixing (left−right diagnostic comparison).
        symmetry_pairs: Optional[List[tuple]] = None,
        # G: MedWConcept — [dim, llm_dim] disease-anchor directions.
        #    Re-inits W_concept to align FHE features to LLM disease-concept directions.
        disease_anchor_directions: Optional[torch.Tensor] = None,
        disease_anchor_alpha: float = 0.3,
        # H: MedICMC — [R] per-region magnitude salience.
        #    Per-region β in MagnitudeCalibratedNorm (lung amplify, cardiac suppress).
        region_salience: Optional[torch.Tensor] = None,
        # === v165 CADN: Client-Assisted Dynamic Normalization (historical baseline) ===
        # When True, replaces PolyNorm inside blocks AND CipherRMSNorm final_norm
        # with CADNLayer (EXACT LayerNorm via client callback, depth 0).
        # CADN supersedes PolyNorm: no polynomial approximation, dynamic
        # (per-input mean/var), depth-0 (client provides fresh z_norm ciphertext).
        # This is the third instance of the CA-FHE paradigm (after CANR and CAS).
        # When False (default, backward compat with v164), uses PolyNorm + CipherRMSNorm.
        # See cadn_norm.py and canr.py::cadn_refresh() for the full first-principles analysis.
        # v165 DEPTH IMPACT: per-block depth drops from 5→4 (with LTM, saves 1) or
        #   4→3 (core, saves 1); final_norm 1→0 (saves 1). The normalization component
        #   (PolyNorm→CADN) saves 2 depth (polyval depth 2 → 0), but per-block TOTAL
        #   only saves 1 due to parallel matmul depth overlap.
        #   CADN + degree-1 PolyAct (LPPN) = ZERO cipher×cipher in the block (minimum noise).
        # CAS is NOT needed when CADN is enabled (CADN eliminates PolyNorm's c2·x² term).
        use_cadn: bool = False,
        # Linear PolyNorm is used by the callback-free profile to fit the
        # native CKKS prime-count limit. Historical interactive defaults stay
        # degree-2.
        poly_norm_degree: int = 2,
    ):
        super().__init__()
        if block_type not in ("med_fhe_vit", "med_fhe_vit_v2", "med_fhe_vit_v3"):
            raise ValueError(
                f"Unsupported block_type={block_type!r}. Supported types (v116-v124 only): "
                "'med_fhe_vit' (v116, PolyNorm + Hadamard), "
                "'med_fhe_vit_v2' (v117, AAPN — per-region PolyNorm coeffs), "
                "'med_fhe_vit_v3' (v118+, retained historical baseline — AAPN + PolyAct). "
                "v110/v111/v114/v115 blocks were DELETED in v125 cleanup. "
                "See docs/handoff/STATUS.md for the current evidence."
            )

        self.dim = dim
        self.num_layers = num_layers
        self.num_patches = num_patches
        self.spectral_bound = spectral_bound
        self.grid_size = int(math.sqrt(num_patches))
        self.block_type = block_type
        self.ffn_spectral_bound = ffn_spectral_bound
        # v165 CADN: Client-Assisted Dynamic Normalization flag
        self.use_cadn = use_cadn
        if poly_norm_degree not in (1, 2):
            raise ValueError("poly_norm_degree must be 1 or 2")
        self.poly_norm_degree = poly_norm_degree
        # v135: Linear Token Mixing flag
        self.enable_token_mix = enable_token_mix
        self.fuse_linear_poly_act = bool(fuse_linear_poly_act)
        self.fuse_linear_poly_norm = bool(fuse_linear_poly_norm)
        self.fuse_linear_poly_norm_requested = bool(fuse_linear_poly_norm)
        self.channelwise_polynorm = bool(channelwise_polynorm)
        self.fuse_final_affine = bool(fuse_final_affine)
        self.anatomical_skip_training = False
        self.token_mix_spectral_bound = token_mix_spectral_bound
        # v125/v127/v129/v143/v144: pool_type for MedicalConceptAdapter
        if pool_type not in ("global", "region", "spatial_upsample",
                             "perceiver_decoder", "direct"):
            raise ValueError(
                f"pool_type must be 'global', 'region', 'spatial_upsample', "
                f"'perceiver_decoder', or 'direct', got {pool_type!r}"
            )
        self.pool_type = pool_type
        # v127 SBA: store grid sizes
        self.student_grid = student_grid
        self.target_grid = target_grid
        # v119: PolyAct degree (stored for diagnostics, passed to blocks)
        if poly_act_degree not in (1, 2):
            raise ValueError(f"poly_act_degree must be 1 or 2, got {poly_act_degree!r}")
        self.poly_act_degree = poly_act_degree
        # v120: Depth-adaptive degree schedule (per-block degree override)
        # Resolve the schedule: None → uniform, "auto" → heuristic, list → as-is
        if poly_act_degree_schedule == "auto":
            # v124 LPPN: FHE-aware automatic schedule based on depth and noise budget
            if num_layers <= 3:
                # Shallow: all degree=1 (v124 LPPN historical setting, cos=0.999711±0.000025, N=5)
                self.poly_act_degree_schedule = [1] * num_layers
            elif num_layers <= 6:
                # Medium: first block degree=2 (feature formation), rest degree=1
                self.poly_act_degree_schedule = [2] + [1] * (num_layers - 1)
            else:
                # Deep: all degree=1 (noise budget tight)
                self.poly_act_degree_schedule = [1] * num_layers
        elif poly_act_degree_schedule is not None:
            # Explicit schedule — validate length and values
            if len(poly_act_degree_schedule) != num_layers:
                raise ValueError(
                    f"poly_act_degree_schedule length {len(poly_act_degree_schedule)} "
                    f"!= num_layers {num_layers}"
                )
            for d in poly_act_degree_schedule:
                if d not in (1, 2):
                    raise ValueError(f"poly_act_degree_schedule entries must be 1 or 2, got {d!r}")
            self.poly_act_degree_schedule = list(poly_act_degree_schedule)
        else:
            # None: use uniform poly_act_degree for all blocks
            self.poly_act_degree_schedule = [poly_act_degree] * num_layers

        # Patch embedding (spectral-bounded)
        self.patch_proj = SpectralBoundedLinear(
            in_channels * patch_size * patch_size, dim, bias=True,
            spectral_bound=spectral_bound
        )
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Blocks: v116 (med_fhe_vit) / v117 (med_fhe_vit_v2) / v118+ (med_fhe_vit_v3)
        blocks = []
        for i in range(num_layers):
            if block_type == "med_fhe_vit":
                # v116: MedFHEBlock — DISRUPTIVE medical FHE-native ViT
                # Designed FROM FHE primitives (not FHE-ized ViT):
                #   - PolyNorm: polynomial normalization (depth 2, avoids 1/sqrt singularity)
                #   - PolyMLP with Hadamard activation: σ(x)=(W_a x)⊙(W_b x) (depth 3)
                #   - SkipInit residual (no norm in residual, α baked into W_o)
                # Block depth: 5 (PolyNorm 2 + PolyMLP 3)
                # CANR every 2 blocks: 2×5=10 < 13 budget (3 margin) — 50% reduction
                skip_alpha_init = 1.0 / math.sqrt(max(num_layers, 1))
                block = MedFHEBlock(
                    dim=dim,
                    num_tokens=num_patches,
                    mlp_ratio=mlp_ratio,
                    spectral_bound=spectral_bound,
                    input_range=input_range,
                    noise_base_rate=noise_base_rate,
                    noise_depth_growth=noise_depth_growth,
                    noise_warmup_epochs=noise_warmup_epochs,
                    enable_noise_training=enable_noise_training,
                    dropout=dropout,
                    ffn_spectral_bound=ffn_spectral_bound,
                    skip_alpha_init=skip_alpha_init,
                    poly_norm_degree=poly_norm_degree,
                    num_regions=1,       # v116: shared coeffs (no AAPN)
                    # v165 CADN: Client-Assisted Dynamic Normalization
                    use_cadn=use_cadn,
                    # v156 medical priors (A, E, F)
                    medical_prior=medical_prior,
                    med_poly_act_prior=med_poly_act_prior,
                    symmetry_pairs=symmetry_pairs,
                    enable_token_mix=enable_token_mix,  # v135 LTM
                    token_mix_spectral_bound=token_mix_spectral_bound,  # v135 LTM
                    fuse_linear_poly_act=fuse_linear_poly_act,
                    fuse_linear_poly_norm=fuse_linear_poly_norm,
                    channelwise_polynorm=channelwise_polynorm,
                )
            elif block_type == "med_fhe_vit_v2":
                # v117: MedFHEBlock with AAPN — Anatomy-Aware Polynomial Normalization
                # v116 base + per-region polynomial coefficients in PolyNorm:
                #   - Each token uses its anatomical region's polyval coeffs
                #   - 5 regions (upper lung, mid lung L, cardiac, lower lung L, lower lung R)
                #   - ZERO extra FHE depth (same polyval, just different coeffs per token)
                #   - AnatomyRouter provides token_regions (FIXED, based on 2D position)
                # Block depth: 5 (SAME as v116 — per-region coeffs are FREE)
                skip_alpha_init = 1.0 / math.sqrt(max(num_layers, 1))
                block = MedFHEBlock(
                    dim=dim,
                    num_tokens=num_patches,
                    mlp_ratio=mlp_ratio,
                    spectral_bound=spectral_bound,
                    input_range=input_range,
                    noise_base_rate=noise_base_rate,
                    noise_depth_growth=noise_depth_growth,
                    noise_warmup_epochs=noise_warmup_epochs,
                    enable_noise_training=enable_noise_training,
                    dropout=dropout,
                    ffn_spectral_bound=ffn_spectral_bound,
                    skip_alpha_init=skip_alpha_init,
                    poly_norm_degree=poly_norm_degree,
                    num_regions=5,       # v117 AAPN: 5 anatomical regions
                    activation_type='hadamard',  # v117 uses Hadamard
                    # v165 CADN: Client-Assisted Dynamic Normalization
                    use_cadn=use_cadn,
                    # v156 medical priors (A, E, F)
                    medical_prior=medical_prior,
                    med_poly_act_prior=med_poly_act_prior,
                    symmetry_pairs=symmetry_pairs,
                    enable_token_mix=enable_token_mix,  # v135 LTM
                    token_mix_spectral_bound=token_mix_spectral_bound,  # v135 LTM
                    fuse_linear_poly_act=fuse_linear_poly_act,
                    fuse_linear_poly_norm=fuse_linear_poly_norm,
                    channelwise_polynorm=channelwise_polynorm,
                )
            else:  # block_type == "med_fhe_vit_v3"
                # v118/v119/v124: MedFHEBlock with AAPN + PolyAct
                # v117 base + PolyAct activation (replaces Hadamard):
                #   - σ(h) = a0 + a1·h + a2·h² (h = W_a·z) — degree-2 polynomial (v118)
                #   - σ(h) = a0 + a1·h        (h = W_a·z) — degree-1 LINEAR (v119)
                #   - v118 uses TenSEAL polyval (Horner): 1 cipher×plain + 1 cipher×cipher
                #   - v119 uses cipher×plain ONLY (NO cipher×cipher) — eliminates the
                #     dominant correlated CKKS noise source.
                #   - SAVES 1 matmul_plain per block vs Hadamard (W_b eliminated)
                #   - v124 LPPN: [1,1,1] uniform degree=1 historical baseline
                # Block depth: 6 (v118: PolyNorm 2 + W_a 1 + polyval 2 + W_o 1)
                #              5 (v119/v124: PolyNorm 2 + W_a 1 + cipher×plain 1 + W_o 1)
                skip_alpha_init = 1.0 / math.sqrt(max(num_layers, 1))
                block = MedFHEBlock(
                    dim=dim,
                    num_tokens=num_patches,
                    mlp_ratio=mlp_ratio,
                    spectral_bound=spectral_bound,
                    input_range=input_range,
                    noise_base_rate=noise_base_rate,
                    noise_depth_growth=noise_depth_growth,
                    noise_warmup_epochs=noise_warmup_epochs,
                    enable_noise_training=enable_noise_training,
                    dropout=dropout,
                    ffn_spectral_bound=ffn_spectral_bound,
                    skip_alpha_init=skip_alpha_init,
                    poly_norm_degree=poly_norm_degree,
                    num_regions=num_regions,  # v117 AAPN: configurable (default 5)
                    activation_type='polyact',  # v118/v119/v124 PolyAct (vs v117 Hadamard)
                    poly_act_init='gelu_fit_low_noise',
                    # v119/v120/v124: pass per-block poly_act_degree from schedule
                    poly_act_degree=self.poly_act_degree_schedule[i],
                    # v165 CADN: Client-Assisted Dynamic Normalization
                    use_cadn=use_cadn,
                    # v156 medical priors (A, E, F)
                    medical_prior=medical_prior,
                    med_poly_act_prior=med_poly_act_prior,
                    symmetry_pairs=symmetry_pairs,
                    enable_token_mix=enable_token_mix,  # v135 LTM
                    token_mix_spectral_bound=token_mix_spectral_bound,  # v135 LTM
                    fuse_linear_poly_act=fuse_linear_poly_act,
                    fuse_linear_poly_norm=fuse_linear_poly_norm,
                    channelwise_polynorm=channelwise_polynorm,
                )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        # v117/v118: AnatomyAwareTokenRouter — provides FIXED token_regions for AAPN
        # Created for v117 (med_fhe_vit_v2) and v118 (med_fhe_vit_v3) block types.
        # AnatomyRouter computes token_regions based on 2D grid position.
        # v125: token_regions ALSO used by MedicalConceptAdapter when pool_type='region'.
        if block_type in ("med_fhe_vit_v2", "med_fhe_vit_v3"):
            self.anatomy_router = AnatomyAwareTokenRouter(
                dim=dim,
                num_tokens=num_patches,
                grid_size=self.grid_size,
                num_regions=num_regions,  # v121: configurable (default 5)
            )
            # Compute FIXED token_regions and set on each block
            token_regions = self.anatomy_router.compute_token_regions()
            for block in self.blocks:
                block.token_regions = token_regions
                # v135 ASTM: Also set region structure on the token mixer
                # (adaptive hook — ASTM reuses the SAME region assignment as AAPN)
                if block.token_mixer is not None:
                    block.token_mixer.set_token_regions(token_regions, num_regions)
        else:
            self.anatomy_router = None
            # v125 fallback: compute token_regions via a temporary router for
            # v116 block_type (no anatomy_router). Purely positional, no params.
            if pool_type == "region":
                tmp_router = AnatomyAwareTokenRouter(
                    dim=dim,
                    num_tokens=num_patches,
                    grid_size=self.grid_size,
                    num_regions=num_regions,
                )
                token_regions = tmp_router.compute_token_regions()
            else:
                token_regions = None

        # v165 CADN: When use_cadn=True, replace CipherRMSNorm final_norm with
        # CADNLayer (EXACT LayerNorm via client callback, depth 0). This eliminates
        # the affine-only approximation of final_norm — the client computes the
        # FULL LayerNorm (dynamic mean/var + γ/β affine) in plaintext.
        # When use_cadn=False (v164 backward compat), keep CipherRMSNorm (affine-only).
        if use_cadn:
            self.final_norm = CADNLayer(dim, eps=1e-6, use_mean_subtraction=True)
        else:
            self.final_norm = CipherRMSNorm(dim)

        # v118 END-TO-END FHE PIPELINE: MedicalConceptAdapter integrated into encoder
        # Optional — when llm_dim is provided, adapter projects FHE ViT features
        # to LLM-compatible embeddings under FHE (depth 2: pool + projection).
        # Client decrypts ONLY the final projected features (K tokens × llm_dim,
        # K=1 for global pool, K=num_regions for region pool, K=256 for SBA).
        # Server NEVER sees plaintext features. TRUE end-to-end FHE pipeline.
        # v125: pool_type='region' produces K=num_regions tokens (preserves spatial).
        # v127: pool_type='spatial_upsample' produces K=256 tokens via bilinear
        #   interp (4×4 → 16×16 grid, matches SigLIP native token count for LLM).
        if llm_dim is not None and llm_dim > 0:
            self.concept_adapter = MedicalConceptAdapter(
                dim=dim,
                llm_dim=llm_dim,
                num_concepts=num_concepts if num_concepts is not None else 32,
                num_tokens=num_patches,
                pool_type=pool_type,
                num_regions=num_regions,
                student_grid=student_grid,
                target_grid=target_grid,
                perceiver_config=perceiver_config,
                perceiver_name=perceiver_name,
                # v156 medical priors (C, D, G, H)
                anatomical_prior=anatomical_prior,                    # C
                disease_prior_embeddings=disease_prior_embeddings,    # D
                disease_anchor_directions=disease_anchor_directions,  # G
                disease_anchor_alpha=disease_anchor_alpha,            # G
                region_salience=region_salience,                      # H
            )
            # v125: Set token_regions on adapter for region-aware pooling.
            # token_regions was computed above from anatomy_router (or fallback).
            if pool_type == "region" and token_regions is not None:
                self.concept_adapter.set_token_regions(token_regions)
        else:
            self.concept_adapter = None

        # Global noise simulator (for patch_proj noise, plaintext training only)
        self.global_noise_sim = AdaptiveCKKSNoise(
            base_rate=noise_base_rate,
            depth_growth=noise_depth_growth,
            warmup_epochs=noise_warmup_epochs,
            enabled=enable_noise_training,
        )

    @classmethod
    def shallow_wide(cls, **kwargs) -> "NoiseResilientMAViTEncoder":
        """Build the low-depth, width-preserving callback-free student.

        Two degree-1 blocks with fused affine paths keep CKKS depth small while
        retaining the same patch grid and release interface.  A checkpoint must
        be trained and calibrated for this exact topology; this constructor does
        not convert an existing deep model.
        """
        kwargs.setdefault("num_layers", 2)
        kwargs.setdefault("poly_act_degree", 1)
        kwargs.setdefault("poly_act_degree_schedule", [1, 1])
        kwargs.setdefault("poly_norm_degree", 1)
        kwargs.setdefault("block_type", "med_fhe_vit_v3")
        kwargs.setdefault("use_cadn", False)
        kwargs.setdefault("fuse_linear_poly_act", True)
        kwargs.setdefault("fuse_linear_poly_norm", True)
        kwargs.setdefault("fuse_final_affine", True)
        return cls(**kwargs)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Extract non-overlapping patches from raw image tensor (CLIENT-SIDE).

        Standard ViT patchify: splits [B, C, H, W] into [B, num_patches, patch_dim]
        where num_patches = (H/patch_size)^2 and patch_dim = C * patch_size^2.

        This is a PLAINTEXT operation done on the CLIENT before encryption.
        The server never sees the raw image — only the encrypted patch embeddings.

        Args:
            imgs: [B, C, H, W] image tensor (normalized to [-input_range, input_range]).

        Returns:
            [B, num_patches, C*patch_size^2] patch tensor (row-major over the
            (H/patch_size) × (W/patch_size) patch grid).

        Raises:
            ValueError: If H or W is not divisible by patch_size.
        """
        B, C, H, W = imgs.shape
        p = self.patch_size
        if H % p != 0 or W % p != 0:
            raise ValueError(
                f"Image size ({H}x{W}) must be divisible by patch_size ({p})."
            )
        h_patches = H // p
        w_patches = W // p
        # [B, H, W, C] → [B, h_patches, p, w_patches, p, C] → [B, h_patches, w_patches, p, p, C]
        # → [B, h_patches*w_patches, p*p*C]
        patches = (
            imgs.permute(0, 2, 3, 1)  # [B, H, W, C]
            .reshape(B, h_patches, p, w_patches, p, C)
            .permute(0, 1, 3, 2, 4, 5)  # [B, h_patches, w_patches, p, p, C]
            .reshape(B, h_patches * w_patches, p * p * C)
        )
        return patches

    def set_epoch(self, epoch: int):
        """Update epoch for all noise simulators (warmup scheduling)."""
        for block in self.blocks:
            block.set_epoch(epoch)
        self.global_noise_sim.current_epoch = epoch

    def _apply_pre_cadn_noise(self, ct_list: list, sigma: float) -> list:
        """v175 Pre-CADN Gaussian Masking: add Gaussian noise to ciphertexts (server-side).

        MODEL STEALING DEFENSE (Gaussian masking of weight-bearing signal):
          Without this defense, the client receives Y = X·W in plaintext during
          CADN (it decrypts the exhausted ciphertext to compute LayerNorm). With
          D independent queries, the client can solve the linear system
          W = (X^T X)^{-1} X^T Y and EXACTLY recover the server's weight W
          (verified: recovery error = 5e-14 at N=64 queries, D=64).

          This method adds fresh Gaussian masking noise N(0, sigma^2) to each
          ciphertext in the ENCRYPTED domain (cipher + plain_vector, 0 depth)
          BEFORE the ciphertext is sent to the client. The client then decrypts
          Y+N (not Y), computes LayerNorm(Y+N), and re-encrypts. The masking
          yields a DP-style recovery bound: error ~ sigma * sqrt(D / N_queries).
          The protected asset is the model IP (weights), not training-data
          individuals, hence "Gaussian masking" (not "DP noise"); the config flag
          `pre_cadn_noise_sigma` is retained for API stability.

          LayerNorm robustness: CADN's LayerNorm normalizes by sqrt(var), which
          suppresses small additive noise. Empirically (v175 experiment),
          sigma=0.01 gives cos(FHE_defended, plain) >= 0.9999 (negligible impact)
          while raising the 64-query recovery error from 5e-14 to 0.92.

        Args:
            ct_list: List of TenSEAL CKKS ciphertexts (exhausted, to be sent to client).
            sigma: Masking noise standard deviation. 0.01 recommended (negligible cos impact).
        Returns:
            List of masked TenSEAL ciphertexts (same length, 0 depth added).
        """
        import numpy as np
        # TenSEAL: ct + plain_list requires len(plain_list) == len(ct) (logical
        # size = number of VALUES in the ciphertext = self.dim, NOT slot_count).
        # Each ciphertext holds one token's D-dim feature vector.
        noise_len = self.dim
        noised = []
        for ct in ct_list:
            # Fresh Gaussian noise per ciphertext (independent per query per block)
            noise = (np.random.randn(noise_len) * sigma).tolist()
            # cipher + plain: homomorphic addition, 0 depth (no rescale)
            ct_noised = ct + noise
            noised.append(ct_noised)
        return noised

    def _apply_refresh_action(
        self,
        x: CiphertextTensor,
        action: RefreshAction,
        cadn_callback: Callable,
        cadn_layer: CADNLayer,
        pre_cadn_noise_sigma: float,
    ):
        """Execute one LBR/PBSR CADN action and return (x, z_norm)."""
        if not action.refresh:
            return x, None

        indices, selected = select_ciphertexts(x._cipher, action.token_indices)
        cipher_to_send = selected
        if pre_cadn_noise_sigma > 0:
            cipher_to_send = self._apply_pre_cadn_noise(
                selected, pre_cadn_noise_sigma
            )
        fresh_cts, z_norm_cts = cadn_callback(
            cipher_to_send, *cadn_layer.get_cadn_params()
        )
        merged_x = merge_ciphertexts(x._cipher, indices, fresh_cts)
        # Never bake-affine unrefreshed tokens: that is a multiply on
        # potentially exhausted ciphertexts. Inactive slots in z_norm are
        # unused because the block identity-skips them.
        if action.token_indices is None:
            merged_z = z_norm_cts
        else:
            merged_z = merge_ciphertexts(x._cipher, indices, z_norm_cts)
        self._last_refresh_stats["cadn_round_trips"] += 1
        self._last_refresh_stats["cadn_token_refreshes"] += len(indices)
        return (
            CiphertextTensor(ciphertexts=merged_x, engine=x._engine, depth=0),
            CiphertextTensor(ciphertexts=merged_z, engine=x._engine, depth=0),
        )

    def forward(self, x, client_callback: Optional[Callable] = None,
                cspe_mode: bool = False,
                cas_callback: Optional[Callable] = None,
                cadn_callback: Optional[Callable] = None,
                pre_cadn_noise_sigma: float = 0.01,
                refresh_policy: str = "interactive",
                refresh_controller: Optional[RefreshController] = None):
        """Forward on plaintext (training, with ACNT noise) or ciphertext (inference).

        Args:
            x: Plaintext tensor [B, N, in_ch*ps*ps] OR CiphertextTensor.
               For cipher mode: x must be a CiphertextTensor with N ciphertexts,
               each holding (in_ch*ps*ps) slots (encrypted image patches).
               For CSPE mode: x must be a CiphertextTensor with N ciphertexts,
               each holding `dim` slots (encrypted patch embeddings, NOT raw patches).
            client_callback: REQUIRED for cipher mode (non-CADN path).
                Signature: fn(ct_list: list) -> list (pure CANR refresh).
            cspe_mode: v131 CSPE (Client-Side Patch Embedding) mode.
                If True, skip patch_proj in cipher path (client already did it
                in plaintext). The input x must be encrypted dim-dim tokens
                (NOT raw patches). This decouples input resolution from FHE cost:
                the client can use larger patches (e.g., 56×56) for higher
                resolution input, while the server FHE cost is UNCHANGED
                (only self-attention + MLP + W_concept, same dim as v129).
            cas_callback: v152 CAS (Client-Assisted Squaring) callback.
                Signature: fn(ct_list: list) -> (fresh_cts, fresh_ct_sq).
                When provided, the encoder uses CAS-CANR at block boundaries to
                obtain fresh x² ciphertexts, eliminating PolyNorm's cipher×cipher
                noise (the dominant CKKS noise source). When None, falls back to
                regular CANR (cipher×cipher in PolyNorm, depth +2).
                NOT needed when cadn_callback is provided (CADN eliminates PolyNorm).
            cadn_callback: v165 CADN (Client-Assisted Dynamic Normalization) callback.
                Signature: fn(ct_list, norm_weight, norm_bias) -> (fresh_cts, z_norm_cts).
                When provided (and self.use_cadn=True), the encoder uses CADN-CANR at
                block boundaries to obtain fresh x AND z_norm=LayerNorm(x,γ,β) ciphertexts.
                CADN replaces PolyNorm inside blocks (EXACT LayerNorm, depth 0) and
                CipherRMSNorm final_norm (EXACT LayerNorm, depth 0). This is the v165
                CADN path: no polynomial normalization approximation and zero cipher×cipher (with degree-1
                PolyAct). When None but self.use_cadn=True, falls back to client_callback
                (CANR only, CADNLayer uses baked EMA stats — less accurate, backward compat).
            pre_cadn_noise_sigma: v175 Pre-CADN Gaussian Masking defense (model stealing defense).
                When > 0, the server adds Gaussian noise N(0, sigma^2) to each exhausted
                ciphertext in the ENCRYPTED domain (cipher + plain, 0 depth) BEFORE
                sending it to the client for CADN refresh. The client then decrypts
                Y+N (not Y=X·W), computes LayerNorm(Y+N), and re-encrypts. This provides
                a differential-privacy guarantee for the server's weight W: the client's
                recovery error is ~sigma*sqrt(D/N) where N is the number of queries.
                Combined with rate limiting, this makes model stealing infeasible.
                Default 0.01 (v175: defense ON by default — 92% recovery error at
                N=64 queries, negligible cos impact ≥0.9999). Set 0.0 to disable
                (ablation / backward compat — NOT recommended for deployment).
            refresh_policy: ``interactive`` permits client CANR/CADN callbacks.
                ``none`` executes one continuous server-side FHE circuit and forbids
                all callbacks and client-side patch embedding.

        Cipher path (v116-v124 FHE-Native inference, NON-CADN):
            1. patch_proj: cipher × plain matmul (+1 depth) → [N, dim] features
               [v131 CSPE: SKIPPED — done on client in plaintext]
            2. + pos_embed: cipher + plain (+0 depth)
            3. CANR refresh (+0 depth, resets depth 1 → 0)
               [v131 CSPE: SKIPPED — depth is already 0]
            4. For each block (CANR after every block except last):
                 block(x, x_sq) — block depth 4 (v124 LPPN+CAS) or 5 (v118+CAS)
                 CANR/CAS refresh (resets depth to 0, gets x² for next block)
            5. FHE final_norm: AFFINE-ONLY (γ_f · x + β_f, +1 depth)
            6. CANR refresh (resets depth to 0)
            7. v118 MedicalConceptAdapter (if present):
                 global pool + concept projection (+2 depth) → [1, llm_dim]
            8. Final CANR refresh (resets depth to 0)
            9. Client decrypts ONLY the 1 token × llm_dim projected features.

        Cipher path (v165 CADN interactive FHE inference):
            1. patch_proj: cipher × plain matmul (+1 depth) → [N, dim] features
               [v131 CSPE: SKIPPED — done on client in plaintext]
            2. + pos_embed: cipher + plain (+0 depth)
            3. CADN-CANR: client computes z_norm=LayerNorm(x,γ_0,β_0) → fresh x + z_norm
               (resets depth to 0, z_norm at depth 0 — EXACT LayerNorm for block 0)
            4. For each block (CADN-CANR after every block except last):
                 block(x, z_norm=z_norm) — block depth 3 (CADN 0 + W_a 1 + PolyAct 1 + W_o 1)
                 [with LTM: depth 4 — token_mixer +1]
                 CADN-CANR: client computes z_norm=LayerNorm(x,γ_{i+1},β_{i+1}) → fresh x + z_norm
            5. CADN final_norm: client computes z_norm=LayerNorm(x,γ_f,β_f) → z_norm at depth 0
               (EXACT LayerNorm, replaces affine-only CipherRMSNorm)
            6. v118 MedicalConceptAdapter (if present):
                 projection matmul (+1 depth) → [K, llm_dim]
            7. Final CANR refresh (resets depth to 0)
            8. Client decrypts ONLY the K tokens × llm_dim projected features.
            v165 DEPTH SAVINGS: per-block depth 5→4 (with LTM, saves 1) or 4→3 (core,
            saves 1); final_norm 1→0 (saves 1). TOTAL savings per forward = 2.
            v165 NOISE: ZERO cipher×cipher (CADN + degree-1 PolyAct = all cipher×plain).
        """
        if isinstance(x, CiphertextTensor):
            if not x.is_cipher:
                x = x._plain
            else:
                # === Cipher inference path (real CKKS) ===
                if refresh_policy not in ("interactive", "none"):
                    raise ValueError("refresh_policy must be 'interactive' or 'none'")
                continuous_fhe = refresh_policy == "none"
                if continuous_fhe:
                    if any(callback is not None for callback in (
                        client_callback, cas_callback, cadn_callback
                    )):
                        raise ValueError(
                            "refresh_policy='none' forbids all client callbacks"
                        )
                    if cspe_mode:
                        raise ValueError(
                            "refresh_policy='none' requires encrypted raw patches; "
                            "client-side patch embedding would expose model parameters"
                        )
                    if self.use_cadn:
                        raise ValueError(
                            "refresh_policy='none' requires use_cadn=False because "
                            "CADN decrypts intermediate activations"
                        )
                elif client_callback is None and cadn_callback is None:
                    raise RuntimeError(
                        "Cipher forward requires client_callback (CANR) or "
                        f"cadn_callback (CADN). block_type={self.block_type!r}, "
                        f"use_cadn={self.use_cadn!r}."
                    )
                # v175 AUDIT FIX: concept_adapter's final CANR refresh (see below,
                # `fresh_cts = client_callback(x._cipher)`) is a PURE cipher refresh
                # — it must NOT normalize the projected concept features. CADN would
                # incorrectly apply LayerNorm to the llm_dim output, destroying the
                # projection. So when concept_adapter is present, client_callback is
                # MANDATORY. A CADN-only deployment (cadn_callback without
                # client_callback) is only valid when concept_adapter is None.
                if (
                    not continuous_fhe
                    and self.concept_adapter is not None
                    and client_callback is None
                ):
                    raise RuntimeError(
                        "CADN-only mode (cadn_callback without client_callback) is "
                        "incompatible with concept_adapter: the final CANR refresh "
                        "after concept_adapter.forward_cipher() requires a PURE "
                        "client_callback refresh (CADN would mis-normalize the "
                        "projected llm_dim features). Provide client_callback too."
                    )

                if cspe_mode:
                    # v131 CSPE: patch_proj AND pos_embed already done on CLIENT
                    # (plaintext). Input x is encrypted dim-dim tokens with pos_embed
                    # already added. Skip patch_proj AND _add_pos_embed_cipher.
                    # v152 FIX: Previously _add_pos_embed_cipher was called here
                    # unconditionally, causing pos_embed to be added TWICE in CSPE
                    # mode (client + server). This systematic input mismatch limited
                    # cos to ~0.983. Fix: only add pos_embed on server in non-CSPE mode.
                    pass
                else:
                    # v129/v128/v127: patch_proj done on SERVER (FHE).
                    # 1. Patch projection: cipher × plain matmul (+1 depth)
                    x = (
                        self._patch_project_chunked_cipher(x)
                        if x.raw_patch_layout is not None
                        else self.patch_proj(x)
                    )
                    # 2. Add positional embedding (cipher + plain, +0 depth)
                    x = self._add_pos_embed_cipher(x)

                # === v165 CADN path selection ===
                # CADN path is active when: self.use_cadn=True AND cadn_callback provided.
                # When active, CADN replaces PolyNorm (inside blocks) and CipherRMSNorm
                # (final_norm) with EXACT client-computed LayerNorm at depth 0.
                # CAS is NOT needed in CADN path (CADN eliminates PolyNorm's c2·x² term).
                # When use_cadn=True but cadn_callback is None, falls back to CANR/CAS
                # with CADNLayer's affine-only path (backward compat, less accurate).
                use_cadn_path = self.use_cadn and cadn_callback is not None
                controller = refresh_controller or RefreshController()
                self._last_refresh_stats = {
                    "cadn_round_trips": 0,
                    "cadn_token_refreshes": 0,
                    "lbr_enabled": controller.schedule is not None,
                    "pbsr_enabled": controller.mask is not None,
                }

                # === Initial refresh (CANR / CAS-CANR / CADN-CANR) ===
                # v152 FIX: Merged the two CANR blocks into ONE conditional.
                # v165 CADN: CADN-CANR is ALWAYS needed (even in CSPE mode) to
                # obtain z_norm for block 0. Unlike CANR (which can be skipped in
                # CSPE mode since depth is already 0), CADN must compute z_norm.
                # LBR may skip this trip and use baked affine CADN fallback.
                x_sq = None
                z_norm = None
                if use_cadn_path:
                    x, z_norm = self._apply_refresh_action(
                        x,
                        controller.action_before_block(0),
                        cadn_callback,
                        self.blocks[0].norm1,
                        pre_cadn_noise_sigma,
                    )
                elif cas_callback is not None:
                    # CAS-CANR: get fresh x AND x² (eliminates cipher×cipher in PolyNorm)
                    fresh_cts, fresh_ct_sq = cas_callback(x._cipher)
                    x = CiphertextTensor(
                        ciphertexts=fresh_cts, engine=x._engine, depth=0
                    )
                    x_sq = CiphertextTensor(
                        ciphertexts=fresh_ct_sq, engine=x._engine, depth=0
                    )
                elif not cspe_mode and not continuous_fhe:
                    # Non-CSPE without CAS: need CANR to reset depth after patch_proj.
                    # v165fix: Guard against None client_callback — previously this
                    # branch would crash with `TypeError: 'NoneType' object is not
                    # callable` if forward() was called in cipher mode without a
                    # client_callback, cadn_callback, or cas_callback. This is a
                    # programmer error (the cipher path REQUIRES a refresh mechanism
                    # to reset depth), so we raise a clear RuntimeError instead of a
                    # confusing NoneType crash.
                    if client_callback is None:
                        raise RuntimeError(
                            "NoiseResilientMAViTEncoder.forward (cipher mode): the "
                            "non-CSPE path requires a client_callback to refresh "
                            "ciphertext depth after patch_proj. Provide "
                            "client_callback (CANR), cas_callback (CAS), or "
                            "cadn_callback (CADN). Calling cipher forward without "
                            "any refresh callback is unsupported — depth would grow "
                            "unbounded and break the FHE depth budget."
                        )
                    fresh_cts = client_callback(x._cipher)  # CANR → depth 0
                    x = CiphertextTensor(
                        ciphertexts=fresh_cts, engine=x._engine, depth=0
                    )
                # else: CSPE without CAS/CADN — depth is 0, skip CANR (v152 fix)

                num_blocks = len(self.blocks)
                for i, block in enumerate(self.blocks):
                    action_i = controller.action_before_block(i)
                    if use_cadn_path:
                        if action_i.compute_block:
                            # Only refreshed / diagnostic tokens enter matmul.
                            # LBR identity-skip and PBSR background tokens are
                            # copied through the residual with no multiply.
                            x = block(
                                x,
                                z_norm=z_norm,
                                active_token_indices=action_i.token_indices,
                            )
                    else:
                        # v152 CAS: pass x_sq to block (cipher×plain, no cipher×cipher)
                        x = block(x, x_sq=x_sq)
                    if continuous_fhe:
                        # TenSEAL owns large native ciphertext buffers.  Drop
                        # Python wrappers at block boundaries so a long
                        # callback-free run does not retain dead intermediates.
                        gc.collect()
                    # Refresh after every block (skip on last — redundant with final_norm).
                    # This matches v119/v124 verified schedule (canr_every=1).
                    if (i + 1) < num_blocks and not continuous_fhe:
                        if use_cadn_path:
                            next_action = controller.action_before_block(i + 1)
                            if next_action.refresh:
                                x, z_norm = self._apply_refresh_action(
                                    x,
                                    next_action,
                                    cadn_callback,
                                    self.blocks[i + 1].norm1,
                                    pre_cadn_noise_sigma,
                                )
                        elif cas_callback is not None:
                            # CAS-CANR: get fresh x AND x² for next block's PolyNorm
                            fresh_cts, fresh_ct_sq = cas_callback(x._cipher)
                            x = CiphertextTensor(
                                ciphertexts=fresh_cts, engine=x._engine, depth=0
                            )
                            x_sq = CiphertextTensor(
                                ciphertexts=fresh_ct_sq, engine=x._engine, depth=0
                            )
                        else:
                            fresh_cts = client_callback(x._cipher)  # CANR → depth 0
                            x = CiphertextTensor(
                                ciphertexts=fresh_cts, engine=x._engine, depth=0
                            )
                            x_sq = None

                # === Final norm ===
                if use_cadn_path:
                    # Final-norm CADN stays on: this is the last representation
                    # before FDBR. LBR may have skipped earlier blocks, but the
                    # diagnostic release still gets exact LayerNorm.
                    _x_refreshed, z_final = self._apply_refresh_action(
                        x,
                        controller.action_final_norm(),
                        cadn_callback,
                        self.final_norm,
                        pre_cadn_noise_sigma,
                    )
                    x = z_final if z_final is not None else _x_refreshed
                else:
                    # Non-CADN: AFFINE-ONLY final norm (no variance, no polynomial).
                    # PolyNorm is INSIDE each block; final_norm is just an affine
                    # projection to LLM-compatible space.
                    # Max depth at this point: 0 (after CANR) → +1=1 < 13 budget ✓
                    if self.fuse_final_affine and self.concept_adapter is not None:
                        # The adapter owns the fused final affine; this keeps
                        # the whole release projection at one CKKS level.
                        pass
                    else:
                        x = self._fhe_final_norm_affine(x, x._engine)  # depth +1
                    if not continuous_fhe:
                        fresh_cts = client_callback(x._cipher)  # CANR → depth 0
                        x = CiphertextTensor(
                            ciphertexts=fresh_cts, engine=x._engine, depth=0
                        )

                # v118 END-TO-END FHE PIPELINE: MedicalConceptAdapter
                # If adapter is present, project FHE features → LLM dim under FHE.
                # Depth: 2 (global pool sum/N +1, projection matmul +1).
                # Max total depth: 0 (after final CANR) + 2 = 2 < 13 budget ✓.
                # Client decrypts ONLY the final 1 token × llm_dim (NOT N × D).
                # Server NEVER sees plaintext features. TRUE end-to-end FHE.
                #
                # v128 CSB (pool_type='spatial_upsample'): forward_cipher returns
                # 16 ciphers (NOT 256). The bilinear interp is done CLIENT-SIDE
                # on plaintext (v128 CSB method DEPRECATED; v129 PCSD
                # server_side_perceiver_decode is the canonical replacement).
                # Depth: 1 (matmul only — bilinear interp moved to client).
                # Total cipher ops: 16 matmuls (vs v127 SBA's 16 + ~1800 ops).
                if self.concept_adapter is not None:
                    x = self.concept_adapter.forward_cipher(x)
                    # Final CANR to reset depth before client decryption.
                    if not continuous_fhe:
                        fresh_cts = client_callback(x._cipher)
                        x = CiphertextTensor(
                            ciphertexts=fresh_cts, engine=x._engine, depth=0
                        )
                return x

        # === Plaintext training path with ACNT noise ===
        B, N, D = x.shape
        x_flat = x.reshape(B * N, -1)
        x_proj = self.patch_proj(x_flat)

        # ACNT: inject noise after patch_proj (simulates fresh ciphertext noise)
        x_proj = self.global_noise_sim.inject_noise(x_proj, depth=0, is_cipher_x_cipher=False)

        x = x_proj.reshape(B, N, self.dim)
        # Adaptive pos_embed: slice to match actual token count (supports sub-scale tests)
        x = x + self.pos_embed[:, :N, :]

        active = self._anatomical_active_tokens()
        for block in self.blocks:
            x = block(x, active_token_indices=active)

        # Final norm:
        # v165 CADN (use_cadn=True): STANDARD LayerNorm (dynamic mean/var + γ/β).
        #   This matches the cipher path's CADN callback (client computes the SAME
        #   LayerNorm in plaintext). ZERO train-inference gap by construction.
        # v116-v124 (use_cadn=False): AFFINE-ONLY (γ_f · x + β_f, no variance).
        #   This matches the cipher path's _fhe_final_norm_affine.
        if self.use_cadn:
            x = self.final_norm.forward_plain(x)  # Standard LayerNorm (CADN-matched)
        else:
            gamma_f = self.final_norm.weight
            beta_f = self.final_norm.bias if self.final_norm.bias is not None else torch.zeros(self.dim)
            if not (self.fuse_final_affine and self.concept_adapter is not None):
                x = x * gamma_f + beta_f  # Affine-only (matches _fhe_final_norm_affine)

        # v118 END-TO-END FHE PIPELINE: Apply MedicalConceptAdapter in plain path
        # when present (mirrors cipher path). Returns [B, 1+num_concepts, llm_dim].
        if self.concept_adapter is not None:
            x = self.concept_adapter.forward_plain(x)
        return x

    def _anatomical_active_tokens(self):
        """PBSR diagnostic indices when anatomical-skip training is on."""
        if not getattr(self, "anatomical_skip_training", False) or not self.training:
            return None
        from .pbsr_scheduler import PBSRScheduler

        num_tokens = int(getattr(self, "num_patches", 16) or 16)
        grid = int(num_tokens ** 0.5)
        return PBSRScheduler(num_tokens=num_tokens, grid_size=grid).compute_mask().diagnostic_indices

    def enable_anatomical_skip_training(self, enabled: bool = True):
        """Train with the same identity-skip mask used by PBSR at inference."""
        self.anatomical_skip_training = bool(enabled)
        return self

    def forward_vision_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """Plaintext tokens after blocks and final norm, before FDBR/adapter.

        This is the student representation that FHE later encrypts. Clinical
        equivalence uses it as the no-CKKS student baseline so the only
        remaining gap versus ``pipeline.run`` is encryption noise + refresh
        policy, not a different pooling graph.
        """
        patches = self.patchify(image)
        batch, tokens, _ = patches.shape
        projected = self.patch_proj(patches.reshape(batch * tokens, -1))
        hidden = projected.reshape(batch, tokens, self.dim)
        hidden = hidden + self.pos_embed[:, :tokens, :]
        active = self._anatomical_active_tokens()
        for block in self.blocks:
            hidden = block(hidden, active_token_indices=active)
        if self.use_cadn:
            return self.final_norm.forward_plain(hidden)
        gamma = self.final_norm.weight
        beta = (
            self.final_norm.bias
            if self.final_norm.bias is not None
            else torch.zeros(self.dim, device=hidden.device, dtype=hidden.dtype)
        )
        return hidden * gamma + beta

    def forward_plain_fhe_match(
        self, x: torch.Tensor, use_poly: bool = True,
    ) -> torch.Tensor:
        """Plain forward that EXACTLY mirrors the cipher path.

        Used for FHE verification: cos(cipher, plain_fhe_match) isolates error sources.

        v116-v124 (all supported block types): ALL polynomial coefficients are
        LEARNED (not fitted). The plain path uses the SAME learned polynomial
        operations as the cipher path. use_poly is IGNORED (kept for API compat).
        cos(cipher, this) = PURE CKKS noise (target > 0.99, verified 0.9997 v124).

        Architecture (matches cipher path exactly, in plaintext):
          - patch_proj + pos_embed (same as standard plain path)
          - For each block: block.forward_plain_fhe_match(x, use_poly)
          - final_norm: affine-only (γ_f · x + β_f)

        Args:
            x: [B, N, patch_dim] plaintext tensor (pre-patch-projection)
            use_poly: IGNORED (kept for API compat — all polys are learned in v116+).

        Returns:
            [B, N, D] plaintext tensor (FHE-intended computation)
        """
        if self.block_type not in ("med_fhe_vit", "med_fhe_vit_v2", "med_fhe_vit_v3"):
            raise RuntimeError(
                f"forward_plain_fhe_match only supports block_type='med_fhe_vit', "
                f"'med_fhe_vit_v2', or 'med_fhe_vit_v3', got {self.block_type!r}."
            )

        B, N, _ = x.shape
        x_flat = x.reshape(B * N, -1)
        x_proj = self.patch_proj(x_flat)
        x = x_proj.reshape(B, N, self.dim)
        # Adaptive pos_embed: slice to match actual token count
        x = x + self.pos_embed[:, :N, :]

        # Each block: FHE-matched plain forward
        for block in self.blocks:
            x = block.forward_plain_fhe_match(x, use_poly=use_poly)

        # Final norm:
        # v165 CADN (use_cadn=True): STANDARD LayerNorm (dynamic mean/var + γ/β).
        #   This matches the cipher path's CADN callback exactly — the client
        #   computes the SAME standard LayerNorm in plaintext. cos(cipher, this)
        #   = PURE CKKS noise (no approximation gap).
        # v116-v124 (use_cadn=False): AFFINE-ONLY (γ_f · x + β_f, no variance).
        #   This matches the cipher path's _fhe_final_norm_affine exactly.
        #   We use CipherRMSNorm in affine-only mode (scale=1, no mean subtraction).
        #
        # STATE-SAFETY (v154 refactor): The plain-mirror path must NOT pollute
        # self.final_norm's baked statistics. The real cipher path depends on
        # the EMA-derived _baked_scale/_baked_mean set by bake_statistics().
        # Previously this method overwrote them with ones/None permanently — a
        # latent bug that corrupts subsequent cipher forward calls. We now
        # save/restore the baked state around the affine-only evaluation.
        if self.use_cadn:
            # CADN: Standard LayerNorm (matches client's CADN computation exactly).
            # No baked state to save/restore — CADNLayer.forward_plain uses
            # nn.LayerNorm directly (dynamic mean/var, no EMA baking needed).
            x = self.final_norm.forward_plain(x)
        else:
            _saved_baked = getattr(self.final_norm, '_baked', False)
            _saved_scale = getattr(self.final_norm, '_baked_scale', None)
            _saved_mean = getattr(self.final_norm, '_baked_mean', None)
            try:
                self.final_norm._baked_scale = torch.ones(self.dim)
                self.final_norm._baked_mean = None
                self.final_norm._baked = True
                if not (self.fuse_final_affine and self.concept_adapter is not None):
                    x = self.final_norm(x)  # affine-only
            finally:
                # v175 FIX: Restore ALL baked attributes inside finally,
                # so they're correctly restored even if self.final_norm(x)
                # raises. Previously _baked_scale/_baked_mean were restored
                # AFTER the finally block, leaving them corrupted on exception.
                self.final_norm._baked = _saved_baked
                self.final_norm._baked_scale = _saved_scale
                self.final_norm._baked_mean = _saved_mean

        # v118 END-TO-END FHE PIPELINE: Apply MedicalConceptAdapter in fhe-match path
        # when present (mirrors cipher path exactly). Returns [B, 1+num_concepts, llm_dim].
        if self.concept_adapter is not None:
            x = self.concept_adapter.forward_plain(x)
        return x

    def _add_pos_embed_cipher(self, x: CiphertextTensor) -> CiphertextTensor:
        """Add positional embedding to ciphertexts (cipher + plain, 0 depth).

        pos_embed is [1, N, D]; for each token i, we add pos_embed[0, i, :] (a
        D-dim plain vector) to ciphertext i. This is cipher + plain = 0 depth.
        """
        ct_list = x._cipher
        engine = x._engine
        pos_embed = self.pos_embed.detach().cpu()  # [1, N, D]
        N = len(ct_list)
        # Slice pos_embed to match actual token count (supports sub-scale tests)
        N_max = min(N, pos_embed.shape[1])
        new_cts = []
        for i in range(N):
            if i < N_max:
                pos_i = pos_embed[0, i].reshape(-1).tolist()
            else:
                # Beyond pos_embed range: add zero (no positional info)
                dim = self.dim
                pos_i = [0.0] * dim
            new_cts.append(ct_list[i] + pos_i)  # cipher + plain (0 depth)
        return CiphertextTensor(ciphertexts=new_cts, engine=engine, depth=x.depth)

    def _patch_project_chunked_cipher(self, x: CiphertextTensor) -> CiphertextTensor:
        """Project raw patches split across multiple CKKS ciphertexts.

        Chunk partials are added in the encrypted domain, so this is exactly
        the wide patch projection while respecting the available CKKS slots.
        """
        layout = x.raw_patch_layout
        required = {"token_count", "parts_per_token", "chunk_size", "feature_dim"}
        if not required.issubset(layout):
            raise ValueError(f"invalid raw patch layout; required keys={sorted(required)}")
        token_count = int(layout["token_count"])
        parts = int(layout["parts_per_token"])
        chunk_size = int(layout["chunk_size"])
        feature_dim = int(layout["feature_dim"])
        expected = token_count * parts
        if len(x._cipher) != expected:
            raise ValueError(
                f"raw patch ciphertext count {len(x._cipher)} != expected {expected}"
            )
        weight_all = self.patch_proj.weight.detach().cpu()
        bias = self.patch_proj.bias.detach().cpu() if self.patch_proj.bias is not None else None
        if weight_all.shape[0] != feature_dim:
            raise ValueError(
                f"raw patch feature_dim {feature_dim} != patch projection input {weight_all.shape[0]}"
            )
        projected = []
        for token in range(token_count):
            accumulated = None
            for part in range(parts):
                start = part * chunk_size
                stop = min(start + chunk_size, feature_dim)
                source_index = token * parts + part
                partial = CiphertextTensor(
                    ciphertexts=[x._cipher[source_index]],
                    engine=x._engine,
                    depth=x.depth,
                ).matmul_plain(weight_all[start:stop, :])
                value = partial._cipher[0]
                accumulated = value if accumulated is None else accumulated + value
                # No later operation needs this raw chunk.  Releasing it here
                # keeps the peak native allocation bounded by one token's
                # partials instead of the whole raw-patch batch.
                x._cipher[source_index] = None
            if bias is not None:
                accumulated = accumulated + bias.reshape(-1).tolist()
            projected.append(accumulated)
            if token % 2 == 1:
                gc.collect()
        return CiphertextTensor(
            ciphertexts=projected,
            engine=x._engine,
            depth=x.depth + 1,
        )

    def _fhe_final_norm_affine(self, x: CiphertextTensor, engine) -> CiphertextTensor:
        """v112 AFFINE-ONLY final norm (no variance, no polynomial).

        Computes: y = γ_f · x + β_f  (per-channel affine only)

        This is the v112 replacement for v111's polynomial LayerNorm final norm.
        Since v112 removes variance normalization entirely (Innovation 1), the
        final norm is just an affine transform.

        Depth: 1 (cipher × plain for γ_f scaling, then + plain for β_f)
          Note: cipher × plain consumes +1 depth in TenSEAL auto_rescale.
          The + β_f is cipher + plain = 0 depth.

        This is the LAST ct*plain in the pipeline — no subsequent matmul to
        break (v106 fusion principle satisfied: next consumer is CANR, then
        the LLM adapter which receives fresh depth-0 ciphertexts).
        """
        gamma_f = self.final_norm.weight.detach().cpu().reshape(-1)
        if self.final_norm.bias is not None:
            beta_f = self.final_norm.bias.detach().cpu().reshape(-1)
        else:
            beta_f = torch.zeros(self.dim)

        # y = x * γ_f + β_f  (cipher × plain + plain, +1 depth)
        result = x * gamma_f.tolist()  # +1 depth (cipher × plain)
        result = result + beta_f.tolist()  # 0 depth (cipher + plain)
        return result  # depth 1

    def multiplicative_depth(self) -> int:
        """Max multiplicative depth per CANR segment (for depth budget verification).

        v165 CADN path (use_cadn=True): per-block depth 3 (CADN 0 + W_a 1 +
            PolyAct 1 + W_o 1), final_norm 0 (CADN exact LayerNorm). Total: 3.
            With LTM: 4. With degree-2 PolyAct: 4 (but CADN+degree-1 = ZERO
            cipher×cipher — the evaluated interactive config).
            v165 CADN + LPPN [1,1,1] + LTM: 4 < 13 budget (9 margin).

        v116/v117 (med_fhe_vit / med_fhe_vit_v2, non-CADN): max CANR-segment depth =
            2 blocks × 5 (PolyNorm 2 + PolyMLP 3) = 10 < 13 budget (3 margin).
            Plus final_norm affine (+1) = 11. CANR every 2 blocks (50% reduction).
            For L=1 (smoke test): 1 block × 5 + 1 final_norm = 6 < 13.

        v118/v119/v124 (med_fhe_vit_v3, non-CADN): max CANR-segment depth depends
            on per-block degree. With canr_every=1 (CANR after every block):
              max(block_depths) + 1 (final_norm affine)
            - degree=1 block: depth 5 (PolyNorm 2 + W_a 1 + cipher×plain 1 + W_o 1)
            - degree=2 block: depth 6 (PolyNorm 2 + W_a 1 + polyval 2 + W_o 1)
            v118 (all degree=2): 6 + 1 = 7 < 13 (6 margin)
            v119 (all degree=1): 5 + 1 = 6 < 13 (7 margin)
            v124 LPPN [1,1,1] (all degree=1): 5 + 1 = 6 < 13 (7 margin) — v164 historical setting
        """
        # v165 CADN: CADN replaces PolyNorm (depth 2→0) and final_norm (depth 1→0).
        # Per-block depth: CADN(0) + W_a(1) + PolyAct(deg-1:1 / deg-2:2) + W_o(1).
        # Final norm: CADN(0) — client provides fresh z_norm at depth 0.
        if self.use_cadn:
            if self.block_type in ("med_fhe_vit", "med_fhe_vit_v2"):
                # Hadamard: CADN(0) + max(W_a,W_b)(1) + h_a*h_b(1) + W_o(1) = 3
                # v175 FIX: W_a and W_b are PARALLEL (both consume z, both at
                # depth+1). Previously counted as sequential (+2), over-reporting
                # by 1. The correct sum uses max(W_a, W_b) = 1, giving total 3.
                # This matches med_fhe_vit_block.multiplicative_depth() (also 3).
                block_depth = 3
            else:
                # PolyAct: CADN(0) + W_a(1) + PolyAct(deg) + W_o(1)
                max_deg = max(self.poly_act_degree_schedule)
                block_depth = 1 + (1 if max_deg == 1 else 2) + 1  # = 3 or 4
                if self.fuse_linear_poly_act and max_deg == 1:
                    block_depth -= 1
            # Add LTM depth if enabled (+1)
            if self.enable_token_mix:
                block_depth += 1
            # CADN final_norm is depth 0 (no +1 for final_norm)
            # v170: Include adapter depth (W_concept projection = +1).
            # The adapter projects FHE features → LLM dim under FHE (cipher×plain
            # matmul, depth +1) before client decrypts. Total pipeline depth must
            # include this for accurate budget verification.
            adapter_depth = 0
            if hasattr(self, 'concept_adapter') and self.concept_adapter is not None:
                adapter_depth = self.concept_adapter.multiplicative_depth()
            return block_depth + adapter_depth

        if self.block_type in ("med_fhe_vit", "med_fhe_vit_v2"):
            # v116/v117: per-block depth = 5 (PolyNorm 2 + PolyMLP 3).
            # v124+ uses canr_every=1 (CANR after every block, see forward() line 873),
            # so max CANR-segment depth = 1 block × 5 + 1 final_norm = 6 < 13 budget
            # (7 margin). The old canr_every=2 schedule (2 blocks × 5 = 10) was
            # superseded in v124; this function reflects the v124+ deployed schedule.
            num_blocks_per_canr = 1
            encoder_depth = num_blocks_per_canr * 5 + 1  # +1 for final_norm affine
        else:
            # v118/v119/v120/v124: max CANR-segment depth depends on per-block degree
            max_block_depth = max(
                5 if d == 1 else 6 for d in self.poly_act_degree_schedule
            )
            if self.fuse_linear_poly_act and all(d == 1 for d in self.poly_act_degree_schedule):
                max_block_depth -= 1
            if self._linear_poly_norm_fusion_active():
                max_block_depth -= 1
            encoder_depth = max_block_depth + int(
                not (self.fuse_final_affine and self.concept_adapter is not None)
            )  # final_norm affine unless folded into adapter
        # v170: Include adapter depth
        if hasattr(self, 'concept_adapter') and self.concept_adapter is not None:
            encoder_depth += self.concept_adapter.multiplicative_depth()
        return encoder_depth

    def continuous_multiplicative_depth(self) -> int:
        """Depth of the complete callback-free cipher circuit."""
        from fpavlm.deployment import continuous_fhe_depth

        norm_fusion = self._linear_poly_norm_fusion_active()
        return continuous_fhe_depth(
            self.poly_act_degree_schedule,
            token_mixing=self.enable_token_mix,
            poly_norm_degree=self.poly_norm_degree,
            fuse_linear_poly_act=self.fuse_linear_poly_act,
            fuse_linear_poly_norm=norm_fusion,
            include_final_affine=not (self.fuse_final_affine and self.concept_adapter is not None),
            include_release_projection=self.concept_adapter is not None,
        )

    def _linear_poly_norm_fusion_active(self) -> bool:
        """Return whether affine PolyNorm fusion is valid for loaded weights."""
        if not self.fuse_linear_poly_norm or self.poly_norm_degree != 1:
            return False
        for block in self.blocks:
            if not block.fuse_linear_poly_norm:
                return False
            c2_regions = getattr(block.norm1, "c2_regions", None)
            c2_shared = getattr(block.norm1, "c2", None)
            coeffs = c2_regions if c2_regions is not None else c2_shared
            if coeffs is not None and float(coeffs.detach().abs().max()) > 1e-8:
                return False
        return True

    def bake_statistics(self):
        """Bake statistics for FHE inference.

        v116-v124 (non-CADN): Prepares cipher weights for each block (v106 fusion +
        SkipInit baking + learned poly coefficient baking) AND bakes
        MedicalConceptAdapter (if present) for end-to-end FHE pipeline.
        Also overrides final_norm to affine-only mode (no variance).

        v165 CADN: Same block preparation, but final_norm is NOT overridden to
        affine-only (CADN computes EXACT LayerNorm via client callback). The
        CADNLayer's bake_statistics() bakes EMA stats for the backward-compat
        fallback path (used only if cadn_callback is None at inference).
        """
        for block in self.blocks:
            block.bake_statistics()
        if self.fuse_linear_poly_norm:
            self.fuse_linear_poly_norm = all(
                block.fuse_linear_poly_norm for block in self.blocks
            )
        self.final_norm.bake_statistics()

        # v117: Bake anatomy router (computes token_regions if not already done)
        if self.anatomy_router is not None:
            self.anatomy_router.bake_statistics()

        # v118: Bake MedicalConceptAdapter (if present)
        if self.concept_adapter is not None:
            if self.fuse_final_affine and not self.use_cadn:
                self.concept_adapter.fuse_input_affine(
                    self.final_norm.weight.detach(),
                    self.final_norm.bias.detach()
                    if self.final_norm.bias is not None
                    else torch.zeros(self.dim),
                )
            self.concept_adapter.bake_statistics()

        # v116-v124 (non-CADN): Override final_norm to affine-only mode (scale=1, no mean).
        # v165 CADN: Do NOT override — CADN final_norm uses EXACT LayerNorm via callback.
        # The CADNLayer.bake_statistics() above already baked EMA stats for the
        # fallback path (used only if cadn_callback is None at inference).
        if not self.use_cadn:
            D = self.dim
            self.final_norm._baked_scale = torch.ones(D)
            self.final_norm._baked_mean = None
            self.final_norm._baked = True
        return self

    def project_spectral(self):
        """Project weights to spectral bound (for training)."""
        self.patch_proj.project_spectral()
        for block in self.blocks:
            block.project_spectral()
