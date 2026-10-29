"""PerceiverDecoder: Server-Side Non-Linear Decoder for FHE-VLM (v129 PCSD, v149+).

v129 INNOVATION: Privacy-Computation Splitting for Non-Linear Decoders (PCSD).
v149+ DEPLOYMENT: Server-side with PRIVATE weights (0 shared model parameters).
Privacy: the client applies the governed DPCFPAccountant release to decrypted
features before sending them to the server. The Perceiver sees only that release.

FIRST-PRINCIPLES ANALYSIS (why v128 reports are still identical):

  v128 CSB produces 256 tokens via BILINEAR INTERPOLATION of 16 input tokens.
  Bilinear interp is LINEAR: out = W_up @ P, where W_up ∈ R^{256×16}, P ∈ R^{16×D}.
  By rank inequality: rank(out) ≤ min(rank(W_up), rank(P)) ≤ min(16, 16) = 16.

  So v128's 256 output tokens span AT MOST 16 dimensions in 2560-dim space.
  Real SigLIP 256 tokens span 256 dimensions (full rank, each from a unique
  896×896 image patch). The LLM detects this rank-16 deficit via low-rank
  attention pattern and falls back to text-only prior ("no findings"), producing
  IDENTICAL reports regardless of pathology (TF-IDF=1.0).

  Empirically verified by v128 experiment (2026-07-21):
    cos(FHE, plain) = 0.9997 (PASS — FHE matches plain perfectly)
    TF-IDF(FHE, GT) = 0.079 (FAIL — reports don't match ground truth)
    reports_differentiated = False (CRITICAL FAIL — all 5 reports identical)

  ROOT CAUSE: rank-16 bottleneck inherent to LINEAR FHE adapters.

v129 SOLUTION: Break the rank-16 bottleneck with NON-LINEAR operations.

  Non-linear operations (softmax, GELU) under FHE are prohibitively expensive
  (each non-linear layer adds 3-5 multiplicative depth, requires polynomial
  approximation, takes hours per image — verified by v127 SBA's 60+ min/image
  for LINEAR bilinear interp alone).

  FIRST-PRINCIPLES INSIGHT (v129, refined v149): The Perceiver cross-attention
  runs on PLAINTEXT (the 16x2560 features the client decrypts and sends to the
  server). ANY architecture is allowed on plaintext. The Perceiver uses softmax
  (non-polynomial) which is infeasible under FHE but trivial in plaintext.

  v129 uses a Perceiver-style cross-attention decoder:
    256 learnable position queries attend to 16 input tokens via cross-attention.
    The softmax in attention is NON-LINEAR, breaking the rank-16 bottleneck.
    Output rank can reach 256 (full rank in 2560-dim space).

v149+ DEPLOYMENT DECISION (server-side, PRIVATE weights):
  The Perceiver (19M learnable params) runs on the SERVER in plaintext, NOT the
  client. This satisfies the constraint "server doesn't share model parameters"
  — the client receives 0 decoder weights. The client only sends 16 decrypted
  (and CFP-noised, v154) features to the server.

  v154 CFP (extended by v157+): The client adds calibrated Gaussian noise to
  the 16x2560 features before sending. This is REQUIRED for privacy: without
  CFP, a learned linear attack achieves 20/20 patient identification from the
  plaintext features. With the historical isotropic mechanism at sigma=8x, worst-case
  identification drops to 10% (≤ 2x chance level). v157 path-mismatch fix made
  features MORE invertible (ridge_cos 0.9568 vs v154's 0.8447), so v157+ uses
  ANISOTROPIC CFP (Innovation B) at sigma=24x, achieving 5% identification =
  chance level (deterministic privacy GUARANTEE). The Perceiver + LLM run on
  noised features. See ARCHITECTURE.md §7.2 for the v157 recalibration.

PRIVACY ANALYSIS (v149+ server-side + v154/v157 CFP):

  WITHOUT CFP (v149-v153): The server sees 16x2560 plaintext features. A
  learned linear attack (ridge regression R^1024 -> R^150528) achieves 20/20
  patient identification — this IS a privacy leak (the features carry patient-
  specific signal, even though the raw image never left the client).

  WITH CFP (v154 sigma=8x isotropic): worst-case identification 10% (≤ 2x chance).
  WITH CFP (v157+ sigma=24x anisotropic B): worst-case identification 5% =
  chance level in that finite attack experiment; this is not a privacy guarantee.

  The FHE chain covers: encoder (3 layers) + W_concept projection. It ends
  BEFORE the Perceiver (softmax is non-polynomial). The server sees only
  CFP-noised plaintext features, which still require stronger attack evaluation.

ARCHITECTURE COMPARISON:

  v128 CSB (linear, rank-16, client-side public decoder):
    Server: 16 ciphers → 16 projected ciphers (depth 1, 16 matmuls)
    Client: decrypt 16 → bilinear interp (1 linear matmul) → 256 tokens (rank≤16)

  v129 PCSD v149+ (non-linear, rank-256, SERVER-side PRIVATE decoder + v154/v157 CFP):
    Server: 16 ciphers → 16 projected ciphers (depth 1, 16 matmuls)  ← SAME
    Client: decrypt 16 → CFP noise (sigma=24x anisotropic B, v157+) → send 16 noised features
    Server: Perceiver cross-attention (PRIVATE) → 256 tokens (rank≤256)  ← NEW

  FHE cost: IDENTICAL (server-side FHE computation unchanged from v128)
  Client cost: ~0s (CFP noise is a single torch.randn, trivial)
  Privacy: v154 CFP (sigma=8x isotropic) gave ≤ 2x chance identification;
    v157+ anisotropic CFP (sigma=24x) reached chance-level identification in the
    historical multi-seed attack, but corrected DP accounting is weak.

PARAMETERS (default config):
  - Position queries: 256 × 2560 = 655K
  - Q projection: 2560 × 512 = 1.3M
  - K projection: 2560 × 512 = 1.3M
  - V projection: 2560 × 512 = 1.3M
  - Output projection: 512 × 2560 = 1.3M
  - LayerNorm × 2: 10K
  - FFN (2560 → 2560 → 2560): 13.1M
  - Total: ~19M params (38MB fp16, fits easily on 4070 8GB)

References:
  - Perceiver: Jaegle et al. "Perceiver: General Perception with Iterative
    Attention" (ICML 2021). arXiv:2103.03206
  - Perceiver Resampler: Alayrac et al. "Flamingo: a Visual Language Model
    for Few-Shot Learning" (NeurIPS 2022).
  - Feature inversion: Mahendran & Vedaldi (CVPR 2015). arXiv:1412.0035
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# v167: Plug-and-play perceiver registry
from .base_perceiver import BasePerceiverDecoder, PerceiverRegistry
from .identity_perceiver import IdentityPerceiver  # noqa: F401 (registers "identity")


class MagnitudeCalibratedNorm(nn.Module):
    """v136 MCPO: Magnitude-Calibrated Normalization — preserves token-wise
    magnitude diversity that LayerNorm destroys.

    ROOT CAUSE of LLM ignoring features (v135 diagnostic, 2026-07-22):
      The Perceiver's output LayerNorm normalizes each token to mean=0, std=1,
      DESTROYING the token-wise magnitude diversity that the LLM needs to
      distinguish informative from background tokens.

      Diagnostic evidence (v135, 5 OpenI images):
        - Student token norm CV = 0.0176 (all tokens ~32.6, near-identical)
        - Teacher token norm CV = 0.1452 (tokens range 15-42, diverse)
        - Student cross-image cosine = 0.9998 (pathologies indistinguishable!)
        - Teacher cross-image cosine = 0.97-0.99 (pathologies distinguishable)
        - LLM produces identical "clear lungs" reports for all 5 pathologies
        - TF-IDF(Ref, GT) = 0.076 (reports don't match ground truth)

      WHY this happens: SigLIP (teacher) produces tokens where informative
      patches (pathology) have HIGHER magnitude than background patches. The
      LLM's attention mechanism uses this magnitude as a "relevance signal" —
      high-magnitude tokens get more attention, low-magnitude tokens are
      suppressed. LayerNorm flattens ALL tokens to std=1, destroying this
      signal. The LLM then sees all 256 tokens as equally "uninformative"
      and falls back to the text-only prior ("no findings").

    v136 SOLUTION: LayerNorm (for training stability) + learnable per-token
      magnitude scale (to restore magnitude diversity):

        out[t] = LayerNorm(x[t]) * exp(log_scale[t])

      where log_scale[t] is a learnable per-token parameter (initialized to 0,
      so exp(0)=1.0 = identity at init). The model learns which tokens should
      be "active" (high magnitude, informative regions) and which should be
      "background" (low magnitude, normal tissue).

      The exp() ensures the scale is always positive, and log-space
      parameterization provides symmetric gradients around 0.

    v136 FAILED: The per-token scale is IMAGE-INVARIANT. It can encode a
      spatial prior ("center is usually informative") but CANNOT produce
      image-specific patterns ("THIS image has pathology in lower-left").
      Result: all images produce the same magnitude pattern → LLM ignores
      features → TF-IDF=0.0747, all reports identical.

    v138-v139 FAILED: MagnitudePreservingLayerNorm (MPLN) tried to preserve
      magnitude THROUGH the LayerNorm itself. But this creates a CONFLICT
      between direction (cosine loss wants LayerNorm) and magnitude (diversity
      loss wants preservation). The model either rejected preservation (v138:
      alpha<0) or sacrificed directional accuracy (v139b: val_cos 0.82→0.75).
      Additionally, v139b's CV=0.70 was 5x the teacher's 0.145 — the LLM
      couldn't interpret features with wrong magnitude distribution.

    v140 SOLUTION: Image-Conditioned Magnitude Calibration (ICMC).
      SEPARATE direction from magnitude — don't fight within the LayerNorm.

      1. STANDARD LayerNorm (best directional accuracy, val_cos≈0.82)
      2. MCPO per-token scale (image-INVARIANT spatial prior, from v136)
      3. ICMC: image-CONDITIONED magnitude from input token norms

      ICMC routes the 16 input tokens' magnitude pattern to 256 output tokens
      via BILINEAR INTERPOLATION (reusing CSPE's spatial upsampling principle):

        input_mags = input_tokens.norm(dim=-1)           # [B, 16] image-specific
        mag_2d = input_mags.view(B, 1, sg, sg)            # spatial layout
        mag_up = bilinear_interp(mag_2d, (tg, tg))        # [B, 1, tg, tg]
        pattern = mag_up.flatten() / mean(mag_up)         # [B, 256], mean=1
        out = LayerNorm(x) * exp(log_scale) * (1 + β * (pattern - 1))

      At β=0: v136 behavior (MCPO only, no image-conditioning) — warm-start safe
      At β>0: image-specific magnitude pattern applied on top of MCPO

      KEY DIFFERENCE from v137 MH: v137 routed magnitude through SOFTMAX
      attention weights (probability distribution → weighted AVERAGE → reduced
      variance, too smooth). v140 ICMC uses BILINEAR INTERPOLATION (direct
      spatial upsampling → preserves the sharp spatial pattern from 16 tokens).

      KEY DIFFERENCE from v138/v139 MPLN: MPLN modified the LayerNorm itself
      (conflict between direction and magnitude). ICMC operates AFTER LayerNorm
      (direction and magnitude are SEPARATED — no conflict).

      CALIBRATION: The model learns β to match the teacher's magnitude level.
      Input tokens have CV≈0.74, teacher output has CV≈0.145. At β≈0.2, the
      ICMC pattern's contribution gives CV≈0.145 (matching teacher). The MSE
      and magnitude_pattern losses provide gradient to find the right β.

    ADAPTIVE INNOVATION (architecture-fit, not application-level):
      - EXTENDS the existing MCPO (adds image-conditioning to existing scale)
      - REUSES the existing bilinear interpolation from CSPE (same spatial
        upsampling principle, F.interpolate bilinear mode)
      - FOLLOWS SkipInit pattern (β=0 at init, same as encoder skip_alpha)
      - SEPARATES direction (LayerNorm) from magnitude (ICMC) — no conflict
      - Client-side (plaintext, NO FHE cost)
      - The MCPO scale captures image-INVARIANT spatial prior; ICMC captures
        image-SPECIFIC magnitude pattern. Together: complete magnitude info.

    PRIVACY: UNCHANGED — ICMC uses only PUBLIC decoder weights and operates
      on plaintext (input token norms, after decryption). No FHE cost.

    PARAMETERS: num_tokens (256 log_scale) + 1 scalar (icmc_beta) — negligible.

    v164/v165 AMPLIFICATION: BETA_EFFECT_SCALE amplifies the MedICMC per-region β
      effect (Innovation H). ROOT CAUSE: β operates on (mag_pattern - 1) which
      is a zero-mean deviation (typically ±0.3), so even β~0.2 gives only
      ~6% local change — below the 1% feature-norm threshold after averaging.
      BETA_EFFECT_SCALE=20.0 amplifies both the forward effect AND the gradient
      (chain rule), so β receives a 20× stronger learning signal. The LayerNorm
      direction is UNCHANGED (β only affects magnitude, post-norm). No FHE
      impact (this runs on plaintext server-side after FHE chain ends).

      v165fix: scale was raised 5.0 → 20.0 (4× increase) to compensate for CADN's
      exact LayerNorm, which normalizes each token to mean=0/var=1 and reduces the
      (mag_pattern - 1) deviation that H's β operates on (see inline comment below).
    """

    # v164/v165: Amplification scale for MedICMC per-region β (Innovation H).
    # See class docstring for first-principles justification.
    #
    # v165fix: Increased from 5.0 → 20.0 to compensate for CADN's exact LayerNorm.
    # ROOT CAUSE: CADN (v165) normalizes each token to mean=0, var=1, making all
    # token norms ≈ sqrt(D)·γ (uniform). This reduces the (mag_pattern - 1) deviation
    # that H's β operates on, weakening H's effect from 3.08% (v164 PolyNorm, scale=5.0)
    # to 0.64% (v165 CADN, scale=5.0) — below the 1% contribution threshold (INERT).
    # PolyNorm (v164) doesn't fully normalize, so token norms retain variation, keeping
    # H active at scale=5.0.
    #
    # FIX: Increase BETA_EFFECT_SCALE 4× (5.0→20.0) to amplify the reduced (mag_pattern-1)
    # signal under CADN. Expected: H rel_L2 ≈ 0.64% × 4 = 2.56% (comfortably > 1%).
    # This is consistent with the project's established amplification-scale approach
    # (see project_memory: "SkipInit parameters need AMPLIFICATION SCALES to measurably
    # contribute"). The scale is a CLASS CONSTANT (not learned), applied identically in
    # train+inference (no gap). FHE depth UNCHANGED (scalar mul is magnitude-agnostic).
    BETA_EFFECT_SCALE: float = 20.0

    def __init__(self, num_tokens: int, llm_dim: int,
                 student_grid: int = 4, target_grid: int = 16,
                 # v156 Innovation H: MedICMC — per-region magnitude calibration
                 region_salience: Optional[torch.Tensor] = None,
                 num_regions: int = 5):
        super().__init__()
        # v140: STANDARD LayerNorm (revert from v138 MPLN) for best directional
        # accuracy. Direction and magnitude are now SEPARATED — LayerNorm handles
        # direction, ICMC handles image-specific magnitude.
        self.layer_norm = nn.LayerNorm(llm_dim)
        # Per-token magnitude scale (image-INVARIANT spatial prior, from v136)
        # Initialized to 0 → exp(0) = 1.0 (identity at init, warm-start safe)
        self.log_token_scale = nn.Parameter(torch.zeros(num_tokens))
        # v140 ICMC: Image-Conditioned Magnitude Calibration
        # Routes 16 input token norms → 256 output tokens via bilinear interp
        # SkipInit: β=0 at init → v136 behavior (no image-conditioning)
        self.student_grid = student_grid
        self.target_grid = target_grid

        # v156 Innovation H: MedICMC — per-region β (lesion-saliency calibration).
        # Replaces scalar icmc_beta with per-region icmc_beta_regions.
        # Lung fields (salience>0): high input norm = potential lesion → amplify (β>0)
        # Cardiac (salience<0): high input norm = normal dense tissue → suppress (β<0)
        # Init: β_r = β_default × region_salience_r (β_default=0.2, v140 calibrated value).
        # When region_salience is None: falls back to v140 scalar β (backward compat).
        if region_salience is not None:
            salience = torch.as_tensor(region_salience, dtype=torch.float)
            assert salience.shape[0] == num_regions, (
                f"region_salience has {salience.shape[0]} regions, expected {num_regions}"
            )
            beta_default = 0.2  # v140 calibrated value
            self.icmc_beta_regions = nn.Parameter(beta_default * salience)
            self._use_med_icmc = True
            self.num_regions = num_regions
            # Precompute output-token → region mapping (256 output tokens → R regions)
            self._output_token_regions: Optional[List[int]] = \
                self._compute_output_token_regions(student_grid, target_grid, num_regions)
            # No scalar beta in MedICMC mode
            self.icmc_beta = None
        else:
            # v140 backward compat: scalar β
            self.icmc_beta = nn.Parameter(torch.tensor(0.0))
            self.icmc_beta_regions = None
            self._use_med_icmc = False
            self.num_regions = 1
            self._output_token_regions = None

    def _compute_output_token_regions(
        self, student_grid: int, target_grid: int, num_regions: int
    ) -> List[int]:
        """v156 Innovation H: Map each of target_grid² output tokens to a region.

        Uses the SAME region assignment as AnatomyRouter (chest X-ray 5-region
        prior): each output position (i,j) in target_grid is mapped to the
        nearest student_grid position, then to a region via the medical prior
        (upper lung / mid-L / cardiac / lower-L / lower-R).

        This ensures the per-region β is applied to the CORRECT anatomical
        region for each of the 256 output tokens.
        """
        from .anatomy_router import AnatomyAwareTokenRouter
        # Create a temporary router with student_grid to get the region mapping
        router = AnatomyAwareTokenRouter(
            dim=1, num_tokens=student_grid * student_grid,
            grid_size=student_grid, num_regions=num_regions,
        )
        student_regions = router.compute_token_regions()  # [student_grid²]
        # Upsample: each target_grid position maps to nearest student_grid position
        output_regions: List[int] = []
        for i in range(target_grid):
            for j in range(target_grid):
                # Nearest student_grid position (align_corners=False convention)
                src_i = int(round((i + 0.5) * student_grid / target_grid - 0.5))
                src_j = int(round((j + 0.5) * student_grid / target_grid - 0.5))
                src_i = max(0, min(student_grid - 1, src_i))
                src_j = max(0, min(student_grid - 1, src_j))
                student_idx = src_i * student_grid + src_j
                output_regions.append(student_regions[student_idx])
        return output_regions

    def forward(self, x: torch.Tensor,
                input_mags: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B, N, D] → [B, N, D] with per-token + image-conditioned magnitude.

        Args:
            x: [B, N, D] features from Perceiver cross-attention
            input_mags: [B, num_input_tokens] per-token norms of the 16 input
                tokens (BEFORE position embeddings). None during warm-start init.
        """
        # 1. Standard LayerNorm (direction normalization, best accuracy)
        x_norm = self.layer_norm(x)  # [B, N, D]
        # 2. MCPO per-token scale (image-INVARIANT spatial prior)
        scale = torch.exp(self.log_token_scale)  # [N]
        out = x_norm * scale.unsqueeze(0).unsqueeze(-1)  # [B, N, D]

        # 3. v140 ICMC: Image-conditioned magnitude pattern
        if input_mags is not None:
            B = x.size(0)
            # Reshape to spatial grid: [B, 1, sg, sg]
            mag_2d = input_mags.view(B, 1, self.student_grid, self.student_grid)
            # Bilinear interpolation to target grid: [B, 1, tg, tg]
            # (reuses CSPE's spatial upsampling principle)
            mag_up = F.interpolate(
                mag_2d, size=(self.target_grid, self.target_grid),
                mode='bilinear', align_corners=False,
            )
            # Flatten: [B, tg*tg] = [B, num_tokens]
            mag_pattern = mag_up.view(B, -1)
            # Normalize to mean=1 (keeps relative pattern, removes absolute scale)
            mag_pattern = mag_pattern / (mag_pattern.mean(dim=-1, keepdim=True) + 1e-8)
            # v156 Innovation H: per-region β (lesion-saliency magnitude calibration)
            # Lung fields (β>0): high input norm = potential lesion → amplify
            # Cardiac (β<0): high input norm = normal dense tissue → suppress
            # v164: BETA_EFFECT_SCALE amplifies the effect (see class docstring).
            beta_scale = self.BETA_EFFECT_SCALE
            if self._use_med_icmc:
                # Gather per-output-token β by region: [num_tokens] → [1, num_tokens, 1]
                region_indices = torch.tensor(
                    self._output_token_regions, device=x.device, dtype=torch.long
                )
                beta_tok = self.icmc_beta_regions[region_indices].view(1, -1, 1)
                out = out * (1.0 + beta_scale * beta_tok * (mag_pattern - 1.0).unsqueeze(-1))
            else:
                # v140 backward compat: scalar β (SkipInit: β=0 → identity)
                out = out * (1.0 + beta_scale * self.icmc_beta * (mag_pattern - 1.0).unsqueeze(-1))

        return out

    def extra_repr(self) -> str:
        if self._use_med_icmc:
            beta_str = f"icmc_beta_regions={self.icmc_beta_regions.tolist()}"
        else:
            beta_str = f"icmc_beta={self.icmc_beta.item():.4f}"
        return (f"num_tokens={self.log_token_scale.numel()}, "
                f"llm_dim={self.layer_norm.normalized_shape[0]}, "
                f"{beta_str}, "
                f"grid={self.student_grid}→{self.target_grid}")


class CrossAttentionLayer(nn.Module):
    """One layer of cross-attention with FFN residual (Perceiver-style).

    The query tokens attend to the key/value tokens (input tokens) via
    multi-head cross-attention, followed by a FFN with GELU activation.

    The softmax in attention is NON-LINEAR — this is what breaks the rank-16
    bottleneck of v128's linear bilinear interpolation.

    Args:
        llm_dim: Feature dimension (e.g., 2560 for MedGemma 4B)
        num_heads: Number of attention heads (default 4)
        attention_dim: Total attention dimension (default 512, head_dim=128)
        ffn_dim: FFN hidden dimension (default = llm_dim for compact size)
        dropout: Dropout rate (default 0.0, set to 0 for inference determinism)
    """

    def __init__(
        self,
        llm_dim: int = 2560,
        num_heads: int = 4,
        attention_dim: int = 512,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        zero_init_o_proj: bool = False,
    ):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim ({attention_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.llm_dim = llm_dim
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // num_heads
        self.ffn_dim = ffn_dim if ffn_dim is not None else llm_dim
        # v141: zero_init_o_proj enables true SkipInit for layer 2+ of the
        # Perceiver. With o_proj.weight=0, the attention branch contributes
        # nothing at init (output=0), so the layer starts as near-identity
        # (residual passes through; norm_attn_out(queries+0) ≈ queries when
        # queries are already LayerNorm-normalized by the previous layer).
        # Combined with the existing zero-init ffn[3], the ENTIRE layer 2 is
        # a near-identity at init → v141 starts from v140/v136's exact behavior.
        # As training progresses, o_proj and ffn[3] grow from zero, adding
        # iterative non-linear refinement (Perceiver IO principle).
        self.zero_init_o_proj = zero_init_o_proj

        # Projections (no bias for cleaner attention)
        self.q_proj = nn.Linear(llm_dim, attention_dim, bias=False)
        self.k_proj = nn.Linear(llm_dim, attention_dim, bias=False)
        self.v_proj = nn.Linear(llm_dim, attention_dim, bias=False)
        self.o_proj = nn.Linear(attention_dim, llm_dim, bias=False)

        # v140 ICMC: Revert ALL norms to STANDARD LayerNorm. Direction and
        # magnitude are now SEPARATED — LayerNorm handles direction (best
        # directional accuracy, val_cos≈0.82), ICMC handles image-specific
        # magnitude (in MagnitudeCalibratedNorm after the cross-attention).
        # v138/v139 FAILED because MPLN created a CONFLICT between direction
        # (cosine loss wants LayerNorm) and magnitude (diversity loss wants
        # preservation) within the SAME LayerNorm operation.
        self.norm_q = nn.LayerNorm(llm_dim)
        self.norm_kv = nn.LayerNorm(llm_dim)
        self.norm_attn_out = nn.LayerNorm(llm_dim)
        self.norm_ffn_out = nn.LayerNorm(llm_dim)

        # FFN with GELU (non-linear, further breaks rank structure)
        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, self.ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ffn_dim, llm_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize weights (xavier for projections, zeros for FFN second layer)
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight)
        # v141 SkipInit: layer 2+ zero-inits o_proj so the attention branch
        # starts as identity (contributes 0 to the residual). Layer 1 keeps
        # xavier (warm-start compatible with v136/v140 checkpoints).
        if self.zero_init_o_proj:
            nn.init.zeros_(self.o_proj.weight)
        else:
            nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.xavier_uniform_(self.ffn[0].weight)
        nn.init.zeros_(self.ffn[3].weight)  # Start FFN as identity (residual)

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, Nq, llm_dim] query tokens (e.g., 256 position queries)
            kv: [B, Nkv, llm_dim] key/value tokens (e.g., 16 input tokens)
            attn_mask: Optional attention mask [B, Nq, Nkv] or [Nq, Nkv]

        Returns:
            [B, Nq, llm_dim] updated query tokens
        """
        B, Nq, _ = queries.shape
        Nkv = kv.size(1)

        # Pre-norm (v140: standard LayerNorm — direction only, magnitude handled
        # by ICMC in MagnitudeCalibratedNorm after the cross-attention layers)
        q_norm = self.norm_q(queries)
        kv_norm = self.norm_kv(kv)

        # Projections: [B, N, llm_dim] → [B, N, attention_dim]
        q = self.q_proj(q_norm)  # [B, Nq, attention_dim]
        k = self.k_proj(kv_norm)  # [B, Nkv, attention_dim]
        v = self.v_proj(kv_norm)  # [B, Nkv, attention_dim]

        # Multi-head reshape: [B, N, H, head_dim] → [B, H, N, head_dim]
        q = q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, H, Nq, Nkv]
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)  # NON-LINEAR (breaks rank bottleneck)
        attn = self.dropout(attn)

        # Attention output: [B, H, Nq, head_dim] → [B, Nq, attention_dim]
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.attention_dim)
        out = self.o_proj(out)  # [B, Nq, llm_dim]

        # Residual + Norm (v140: standard LayerNorm)
        queries = self.norm_attn_out(queries + self.dropout(out))

        # FFN with residual (v140: standard LayerNorm)
        ffn_out = self.ffn(queries)
        queries = self.norm_ffn_out(queries + ffn_out)

        return queries


class DiseasePriorModulation(nn.Module):
    """v155 Innovation D: Disease-Prior Concept Injection.

    Lightweight cross-attention that modulates the 256 Perceiver output tokens
    with K disease-term embeddings from MedGemma. Counteracts the LLM's
    negative-prior bias (over-prediction of "no findings" due to training data
    imbalance) by injecting disease-concept signal into the visual tokens.

    FIRST-PRINCIPLES ANALYSIS:
      Medical LLMs (MedGemma) trained on imbalanced radiology data over-predict
      "no findings" — the text prior dominates over weak visual signal. The
      Perceiver produces 256 visually-grounded tokens, but the LLM's attention
      still biases toward "no findings" because the visual tokens are not
      explicitly aligned to disease concepts.

      Innovation D injects K disease-term embeddings (cardiomegaly, effusion,
      consolidation, pneumothorax, ...) as a disease-concept prior. The 256
      visual tokens ATTEND to these K disease tokens, learning which disease
      concepts are present in the image. This biases the LLM toward considering
      disease concepts, counteracting the negative-prior bias.

    ARCHITECTURE (v158+ warm-start safe — γ init = 0.1, NOT 0):
      - 256 visual tokens (queries) attend to K disease tokens (keys/values)
      - γ (learnable scalar, init=0.1 per v158) gates the modulation
      - out = tokens + γ · cross_attn(tokens, disease_tokens)
      - At init (γ=0.1): small disease-concept signal from epoch 1 (avoids the
        v157 SkipInit cold-start deadlock where γ=0 → D redundant → BCE gradient
        never reaches γ → γ stuck at 0 forever; verified v156: γ=0.000000 after
        3 epochs). v158's γ=0.1 + delta-based BCE breaks the deadlock.
      - At γ>0: disease-concept signal injected (learned during fine-tuning)

    MEDICAL JUSTIFICATION:
      - Disease tokens are FIXED (from MedGemma's disease-term embeddings) —
        the model learns HOW to use them, not what they are
      - The cross-attention is IMAGE-CONDITIONED (queries from visual tokens),
        so the modulation is image-specific, not a fixed bias
      - γ=0.1 init (NOT γ=0) ensures D provides disease signal from epoch 1,
        so the BCE-on-delta auxiliary loss can flow gradient to γ and o_proj.
        Training then learns the optimal disease-prior strength.

    PRIVACY: NONE (plaintext, server-side Perceiver). ZERO FHE cost. The
      disease tokens are PUBLIC medical terminology (not patient data).

    PARAMETERS (compact, attention_dim=256):
      - q_proj: 2560 × 256 = 655K
      - k_proj: 2560 × 256 = 655K
      - v_proj: 2560 × 256 = 655K
      - o_proj: 256 × 2560 = 655K (zero-init for SkipInit)
      - LayerNorm ×2: 5K
      - γ: 1 scalar
      - Total: ~2.6M params (5.2MB fp16, fits easily on 4070 8GB)
    """

    def __init__(
        self,
        llm_dim: int = 2560,
        num_tokens: int = 256,
        disease_prior_embeddings: Optional[torch.Tensor] = None,
        attention_dim: int = 256,
        num_heads: int = 4,
    ):
        """Initialize DiseasePriorModulation.

        Args:
            llm_dim: Feature dimension (e.g., 2560 for MedGemma 4B)
            num_tokens: Number of visual tokens (256, matches Perceiver output)
            disease_prior_embeddings: [K, llm_dim] tensor of disease-term
                embeddings from MedGemma (e.g., cardiomegaly, effusion, ...).
                Registered as a non-trainable buffer (the model learns HOW to
                use them, not what they are).
            attention_dim: Compact attention dimension (default 256, vs 512 in
                CrossAttentionLayer — keeps params low for 4070 8GB)
            num_heads: Number of attention heads (default 4, head_dim=64)
        """
        super().__init__()
        self.llm_dim = llm_dim
        self.num_tokens = num_tokens
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        if attention_dim % num_heads != 0:
            raise ValueError(
                f"attention_dim ({attention_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.head_dim = attention_dim // num_heads

        # Disease tokens (FIXED, from MedGemma — not trainable)
        K = disease_prior_embeddings.shape[0]
        self.num_disease_tokens = K
        self.register_buffer(
            "disease_tokens",
            disease_prior_embeddings.detach().clone().float(),
        )

        # Cross-attention projections (no bias for cleaner attention)
        self.q_proj = nn.Linear(llm_dim, attention_dim, bias=False)
        self.k_proj = nn.Linear(llm_dim, attention_dim, bias=False)
        self.v_proj = nn.Linear(llm_dim, attention_dim, bias=False)
        self.o_proj = nn.Linear(attention_dim, llm_dim, bias=False)

        # v158: γ init = 0.1 (NOT 0.0). v157 used SkipInit (γ=0) which created a
        # cold-start deadlock: at γ=0, D produces identity → classification head
        # learns from visual features alone → D is redundant → BCE gradient never
        # reaches γ → cosine loss pushes γ→0 forever. With γ=0.1 init + delta-based
        # BCE loss, D provides disease signal from epoch 1, and the head MUST use it.
        self.gamma = nn.Parameter(torch.tensor(0.1))

        # LayerNorm for stability
        self.norm_q = nn.LayerNorm(llm_dim)
        self.norm_disease = nn.LayerNorm(llm_dim)

        # v157 FIX: Initialize ALL projections with xavier (including o_proj).
        # Previously o_proj was zero-init'd, creating a SkipInit DEADLOCK:
        #   - gamma=0 AND o_proj=0 → out=0 always
        #   - dL/dγ ∝ out = 0 (zero gradient to γ)
        #   - dL/do_proj ∝ γ = 0 (zero gradient to o_proj)
        # Both parameters stuck at 0 forever → D NEVER learned (verified v156:
        # gamma=0.000000 after 3 epochs). Standard SkipInit (Bachlechner 2020)
        # uses γ=0 with NORMAL branch weights — the branch (o_proj) must produce
        # a non-zero output so γ has a gradient to learn from. With xavier o_proj,
        # `out` is non-zero at init → dL/dγ ≠ 0 → γ can grow → o_proj can also
        # learn (once γ≠0). D's disease-prior cross-attention becomes trainable.
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)
        # SkipInit is carried by gamma=0 alone (NOT o_proj=0).

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Modulate 256 visual tokens with disease-prior signal.

        Args:
            tokens: [B, 256, llm_dim] Perceiver output tokens (after ICMC norm)

        Returns:
            [B, 256, llm_dim] disease-modulated tokens
            out = tokens + γ · cross_attn(tokens, disease_tokens)
        """
        B = tokens.size(0)
        K = self.num_disease_tokens
        dtype = tokens.dtype

        # Project queries (from visual tokens) and keys/values (from disease)
        # v157 FIX: match BOTH dtype AND device (disease_tokens may be on CPU).
        q = self.q_proj(self.norm_q(tokens).to(dtype))  # [B, 256, attn_dim]
        disease = self.disease_tokens.unsqueeze(0).expand(B, -1, -1).to(
            device=tokens.device, dtype=dtype
        )
        k = self.k_proj(self.norm_disease(disease).to(dtype))  # [B, K, attn_dim]
        v = self.v_proj(self.norm_disease(disease).to(dtype))  # [B, K, attn_dim]

        # Multi-head attention: [B, H, N, head_dim]
        q = q.view(B, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: [B, H, 256, K]
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [B, H, 256, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, self.num_tokens, self.attention_dim)
        out = self.o_proj(out)  # [B, 256, llm_dim]

        # v158: Store delta for auxiliary disease-classification loss.
        # The training script applies a classification head to this delta
        # (NOT the full output) so the BCE gradient flows DIRECTLY to gamma.
        # This breaks the cold-start deadlock where the head classifies from
        # visual features alone, making D redundant and gamma stays ~0.
        self._last_delta = self.gamma * out  # [B, 256, llm_dim]

        return tokens + self._last_delta


@PerceiverRegistry.register("v162-default")
class PerceiverDecoder(BasePerceiverDecoder):
    """Server-Side Perceiver Decoder for breaking rank-16 bottleneck (v129 PCSD, v154).

    v167: Registered as "v162-default" in PerceiverRegistry for plug-and-play
    swapping. Inherits BasePerceiverDecoder so MedicalConceptAdapter can accept
    ANY perceiver (IdentityPerceiver for ablation, future anatomy-aware variants,
    etc.) without code changes.

    Takes 16 plaintext CFP-noised tokens × llm_dim (received from the client after
    it decrypts the server's FHE output and applies CFP) and produces 256 plaintext
    tokens × llm_dim (for LLM input).

    Uses Perceiver-style cross-attention: 256 learnable position queries attend
    to 16 input tokens. The softmax non-linearity in attention breaks the
    rank-16 bottleneck of v128's linear bilinear interpolation.

    All operations are on PLAINTEXT (server-side, NOT under FHE — softmax is
    non-polynomial). The decoder weights are PRIVATE (server-owned, NOT shared
    with the client — satisfies the "server does not share model params" constraint).
    The client only sends 16 CFP-noised tokens; it never sees the decoder weights.

    Args:
        llm_dim: LLM input feature dimension (e.g., 2560 for MedGemma 4B)
        num_queries: Number of output query tokens (default 256, matches SigLIP)
        num_input_tokens: Number of input tokens from server (default 16)
        num_heads: Number of attention heads (default 4)
        attention_dim: Total attention dimension (default 512, head_dim=128)
        ffn_dim: FFN hidden dimension (default = llm_dim for compact size)
        num_layers: Number of cross-attention layers (default 1)
        init_from_bilinear: If True, initialize queries as bilinear-interpolated
            positions of learnable input position embeddings. This makes the
            decoder start from v128's linear baseline and learn non-linear
            improvements. (default True)
        student_grid: Student FHE image grid size (default 4, for bilinear init)
        target_grid: Target LLM image grid size (default 16, for bilinear init)
        dim: Alias for llm_dim (BasePerceiverDecoder protocol uses `dim`)
        num_latents: Alias for num_queries (BasePerceiverDecoder protocol uses `num_latents`)
    """

    def __init__(
        self,
        llm_dim: int = 2560,
        num_queries: int = 256,
        num_input_tokens: int = 16,
        num_heads: int = 4,
        attention_dim: int = 512,
        ffn_dim: Optional[int] = None,
        num_layers: int = 1,
        init_from_bilinear: bool = True,
        student_grid: int = 4,
        target_grid: int = 16,
        anatomical_prior: Optional[torch.Tensor] = None,
        disease_prior_embeddings: Optional[torch.Tensor] = None,
        # v156 Innovation H: MedICMC — per-region magnitude calibration
        region_salience: Optional[torch.Tensor] = None,
        num_regions: int = 5,
        # v167: BasePerceiverDecoder protocol aliases (backward compat)
        dim: Optional[int] = None,
        num_latents: Optional[int] = None,
        **kwargs,
    ):
        """Initialize PerceiverDecoder.

        v155 additions:
            anatomical_prior: [num_queries, llm_dim] tensor of per-position
                anatomical prototypes (mean patch features from OpenI for each
                16×16 grid position). When provided, queries are initialized as
                a blend of bilinear interpolation and anatomical prototypes,
                with left-right symmetry applied. This gives the Perceiver a
                medical-meaningful spatial starting point (Innovation C).
            disease_prior_embeddings: [K, llm_dim] tensor of disease-term
                embeddings from MedGemma (e.g., cardiomegaly, effusion, ...).
                When provided, a DiseasePriorModulation module modulates the
                256 output tokens with disease priors, counteracting the LLM's
                negative-prior bias (Innovation D).
        v167 additions (backward-compat aliases):
            dim: Alias for llm_dim (BasePerceiverDecoder protocol). When both
                are provided, llm_dim takes precedence.
            num_latents: Alias for num_queries (BasePerceiverDecoder protocol).
                When both are provided, num_queries takes precedence.
        """
        # v167: Resolve dim aliases (BasePerceiverDecoder uses `dim`/`num_latents`)
        if dim is not None and llm_dim == 2560:
            llm_dim = dim
        if num_latents is not None and num_queries == 256:
            num_queries = num_latents

        # Initialize BasePerceiverDecoder (sets self.dim, self.num_latents, etc.)
        super().__init__(
            dim=llm_dim,
            num_latents=num_queries,
            num_input_tokens=num_input_tokens,
        )
        self.llm_dim = llm_dim
        self.num_queries = num_queries
        self.student_grid = student_grid
        self.target_grid = target_grid

        # Learnable input position embeddings [16, llm_dim]
        # (helps the decoder distinguish the 16 spatial positions)
        self.input_pos_embed = nn.Parameter(torch.zeros(num_input_tokens, llm_dim))
        nn.init.trunc_normal_(self.input_pos_embed, std=0.02)

        # Learnable position queries [256, llm_dim]
        self.queries = nn.Parameter(torch.zeros(num_queries, llm_dim))
        if anatomical_prior is not None:
            # v155 Innovation C: Anatomical Perceiver Queries
            # Blend bilinear spatial structure with anatomical prototypes,
            # then apply left-right symmetry (chest X-ray prior).
            self._init_queries_anatomical(
                student_grid, target_grid, anatomical_prior,
            )
        elif init_from_bilinear:
            self._init_queries_from_bilinear(student_grid, target_grid)
        else:
            nn.init.trunc_normal_(self.queries, std=0.02)

        # Cross-attention layers
        # v141: layer 0 uses xavier o_proj (warm-start compatible with v136/v140
        # checkpoints). Layer 1+ uses zero_init_o_proj=True (SkipInit) so the
        # deeper Perceiver starts as near-identity and learns iterative non-linear
        # refinement. This makes num_layers=2 start from v140's exact behavior.
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                llm_dim=llm_dim,
                num_heads=num_heads,
                attention_dim=attention_dim,
                ffn_dim=ffn_dim,
                zero_init_o_proj=(i > 0),
            )
            for i in range(num_layers)
        ])

        # v140 ICMC: Magnitude-Calibrated output norm with STANDARD LayerNorm
        # (direction) + MCPO per-token scale (image-invariant spatial prior)
        # + ICMC (image-conditioned magnitude from 16 input token norms via
        # bilinear interpolation). Direction and magnitude are SEPARATED —
        # no conflict between cosine loss (wants LayerNorm) and diversity loss
        # (wants image-specific magnitude).
        self.output_norm = MagnitudeCalibratedNorm(
            num_queries, llm_dim,
            student_grid=student_grid,
            target_grid=target_grid,
            # v156 Innovation H: per-region β (lesion-saliency magnitude calibration)
            region_salience=region_salience,
            num_regions=num_regions,
        )

        # v155 Innovation D: Disease-Prior Concept Injection
        # Lightweight cross-attention modulating 256 output tokens with K
        # disease-term embeddings from MedGemma. v158+: γ init=0.1 (NOT 0) →
        # small disease signal from epoch 1, avoiding the v157 SkipInit
        # cold-start deadlock. NONE when no disease embeddings provided
        # (backward compatible).
        if disease_prior_embeddings is not None:
            self.disease_modulation = DiseasePriorModulation(
                llm_dim=llm_dim,
                num_tokens=num_queries,
                disease_prior_embeddings=disease_prior_embeddings,
                attention_dim=256,  # compact (vs 512 in CrossAttentionLayer)
                num_heads=4,
            )
        else:
            self.disease_modulation = None

    def _init_queries_from_bilinear(self, student_grid: int, target_grid: int):
        """Initialize queries as bilinear-interpolated position embeddings.

        This makes the decoder START from v128's linear baseline (queries ≈
        bilinear interp of input positions) and learn non-linear improvements
        via cross-attention. Empirically this converges faster than random init.
        """
        from .medical_concept_adapter import compute_bilinear_upsample_matrix
        # W_up: [target^2, student^2] = [256, 16]
        W_up = compute_bilinear_upsample_matrix(student_grid, target_grid)
        # Initialize input position embeddings (small random)
        input_pos = torch.randn(self.num_input_tokens, self.llm_dim) * 0.02
        # queries[k] = sum_i W_up[k, i] * input_pos[i]
        # This gives queries a bilinear spatial structure
        with torch.no_grad():
            self.queries.copy_(W_up @ input_pos)
        # NOTE: input_pos_embed is a separate learnable parameter (NOT input_pos).
        # The decoder learns its own position embeddings during training.

    def _init_queries_anatomical(
        self,
        student_grid: int,
        target_grid: int,
        anatomical_prior: torch.Tensor,
    ):
        """v155 Innovation C: Initialize queries from anatomical prototypes.

        Blends bilinear spatial structure (v128 baseline) with anatomical
        prototypes (mean patch features at each 16×16 grid position from OpenI
        chest X-rays), then enforces left-right symmetry (chest X-ray bilateral
        prior: normal anatomy is bilaterally symmetric; pathology breaks it).

        Medical motivation: The 256 output query positions correspond to a
        16×16 grid over the chest. Each position has a CHARACTERISTIC
        anatomical meaning (apex vs costophrenic angle vs cardiac silhouette).
        Generic bilinear/random init wastes this structure. By initializing
        queries from real anatomical prototypes, the Perceiver starts from a
        medically-meaningful spatial layout and learns non-linear refinements.

        Args:
            student_grid: Student grid size (e.g., 4 → 16 input tokens)
            target_grid: Target grid size (e.g., 16 → 256 output queries)
            anatomical_prior: [num_queries, llm_dim] = [256, llm_dim] tensor
                of per-position anatomical prototypes (pre-computed by
                calibration: mean SigLIP feature at each 16×16 grid position
                across 100 OpenI chest X-rays).
        """
        from .medical_concept_adapter import compute_bilinear_upsample_matrix

        # Step 1: Bilinear spatial structure (v128 baseline)
        # W_up: [target^2, student^2] = [256, 16]
        W_up = compute_bilinear_upsample_matrix(student_grid, target_grid)
        input_pos = torch.randn(self.num_input_tokens, self.llm_dim) * 0.02
        bilinear_queries = W_up @ input_pos  # [256, llm_dim]

        # Step 2: Blend with anatomical prototypes
        # α=0.5: equal blend. Bilinear provides smooth spatial structure;
        # anatomical prior provides medical-meaningful starting features.
        assert anatomical_prior.shape == (self.num_queries, self.llm_dim), (
            f"anatomical_prior shape {anatomical_prior.shape} != "
            f"({self.num_queries}, {self.llm_dim})"
        )
        alpha = 0.5
        blended = alpha * anatomical_prior.to(bilinear_queries.dtype) + \
            (1.0 - alpha) * bilinear_queries

        # Step 3: Enforce left-right symmetry (chest X-ray bilateral prior)
        # For query position (i, j) in target_grid, mirror is (i, tg-1-j).
        # Contralateral positions share init (normal anatomy is symmetric).
        # Training learns ASYMMETRY (pathology breaks symmetry).
        tg = target_grid
        sym_queries = blended.clone()
        for i in range(tg):
            for j in range(tg // 2):  # left half only; mirror to right
                mirror_j = tg - 1 - j
                pos_idx = i * tg + j
                mirror_idx = i * tg + mirror_j
                avg = 0.5 * (blended[pos_idx] + blended[mirror_idx])
                sym_queries[pos_idx] = avg
                sym_queries[mirror_idx] = avg

        with torch.no_grad():
            self.queries.copy_(sym_queries)
        # NOTE: input_pos_embed is a separate learnable parameter. The decoder
        # learns its own position embeddings during training.

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tokens: [B, num_input_tokens, llm_dim] plaintext tokens
                (decrypted from server's FHE output).

        Returns:
            [B, num_queries, llm_dim] decoded tokens for LLM input.
            Each of 256 output tokens has a UNIQUE spatial position derived
            from non-linear cross-attention to 16 input tokens, with
            image-specific magnitude applied via ICMC (v140: bilinear
            interpolation of 16 input token norms → 256 output tokens,
            SEPARATED from direction which uses standard LayerNorm — avoids
            the direction/magnitude conflict that doomed v138/v139).
        """
        B = input_tokens.size(0)

        # v140 ICMC: Capture input token magnitudes BEFORE position embeddings.
        # These norms encode the image-specific saliency pattern (which of the 16
        # spatial regions are informative). ICMC routes this pattern to the 256
        # output tokens via bilinear interpolation in MagnitudeCalibratedNorm.
        input_mags = input_tokens.float().norm(dim=-1)  # [B, 16]

        # Add input position embeddings (helps decoder distinguish positions)
        # input_tokens: [B, 16, llm_dim], input_pos_embed: [16, llm_dim]
        # v157 FIX: match BOTH dtype AND device (input_pos_embed may be on CPU
        # when the Perceiver is called standalone via server_side_perceiver_decode
        # without .to(DEVICE) on the whole adapter).
        kv = input_tokens + self.input_pos_embed.unsqueeze(0).to(
            device=input_tokens.device, dtype=input_tokens.dtype
        )

        # Expand queries to batch: [B, 256, llm_dim]
        # v157 FIX: match BOTH dtype AND device (queries may be on CPU).
        queries = self.queries.unsqueeze(0).expand(B, -1, -1).to(
            device=input_tokens.device, dtype=input_tokens.dtype
        )

        # Apply cross-attention layers.
        # v140: Standard LayerNorm in all norms — direction only. Image-specific
        # magnitude is applied SEPARATELY by ICMC after the cross-attention,
        # avoiding the direction/magnitude conflict that doomed v138/v139.
        x = queries
        for layer in self.layers:
            x = layer(x, kv)

        # v140 ICMC: Standard LayerNorm (direction) + MCPO scale (spatial prior)
        # + ICMC (image-conditioned magnitude from input_mags). The ICMC pattern
        # is bilinearly interpolated from the 16 input token norms to 256 output
        # tokens, giving each image a UNIQUE magnitude pattern (val_xcos < 0.99).
        x = self.output_norm(x, input_mags)

        # v155 Innovation D: Disease-Prior Concept Injection.
        # Modulate 256 output tokens with K disease-term embeddings from
        # MedGemma. v158+: γ init=0.1 (NOT 0) → small disease signal from epoch 1,
        # avoiding the v157 SkipInit cold-start deadlock (where γ=0 → D redundant →
        # γ stuck at 0 forever). Training learns the optimal disease-prior injection
        # strength to counteract the LLM's negative-prior bias ("no findings"
        # over-prediction).
        if self.disease_modulation is not None:
            x = self.disease_modulation(x)

        return x  # [B, num_queries, llm_dim]

    def _server_side_decode_internal(self, plain_tokens: torch.Tensor) -> torch.Tensor:
        """Apply Perceiver decoder on plaintext tokens (server-side internal helper).

        This is the SERVER-side decode invoked by
        MedicalConceptAdapter.server_side_perceiver_decode() after the client
        decrypts the 16 projected ciphers and sends the 16x2560 plaintext
        (CFP-noised) tokens to the server. The Perceiver weights are PRIVATE
        (server-owned) — the client never sees them.

        v154 rename: this method was originally named `client_side_decode` based
        on v129's framing where the Perceiver was public/client-side. In v149+
        the Perceiver moved SERVER-side with PRIVATE weights (to satisfy "server
        does not share model params"), making the old name misleading. The
        method body is unchanged — only the name + docstring are corrected.

        Args:
            plain_tokens: [B, 16, llm_dim] or [16, llm_dim] decrypted + CFP-noised
                tokens received from the client (output of forward_cipher +
                client decryption + CFP noise injection).

        Returns:
            [B, 256, llm_dim] or [256, llm_dim] — decoded tokens with rank up
            to 256 (vs v128's rank ≤ 16).
        """
        squeeze_batch = False
        if plain_tokens.dim() == 2:
            plain_tokens = plain_tokens.unsqueeze(0)
            squeeze_batch = True
        out = self.forward(plain_tokens)
        if squeeze_batch:
            out = out.squeeze(0)
        return out

    # v167: BasePerceiverDecoder protocol — declares that this perceiver uses
    # medical priors (anatomical_prior, disease_prior_embeddings) which are
    # injected at construction time.
    def supports_medical_prior(self) -> bool:
        """PerceiverDecoder uses anatomical_prior + disease_prior_embeddings
        (Innovations C, D) injected at __init__. Returns True when these are
        active (not None)."""
        return self.disease_modulation is not None
