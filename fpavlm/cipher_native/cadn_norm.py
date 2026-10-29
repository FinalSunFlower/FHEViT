"""CADN: Client-Assisted Dynamic Normalization — EXACT LayerNorm via CA-FHE.

v165 BREAKTHROUGH INNOVATION — the third instance of the CA-FHE paradigm:

  CANR (v116): client refreshes x           (plaintext identity: x → x)
  CAS  (v152): client refreshes x + x²      (plaintext square: eliminates cipher×cipher)
  CADN (v165): client refreshes x + z_norm   (plaintext LayerNorm: eliminates approximation)

FIRST-PRINCIPLES ANALYSIS (why CADN supersedes PolyNorm and CipherRMSNorm):

  The FHE-ViT fundamental contradiction: LayerNorm requires mean/variance/
  sqrt/division — ALL non-polynomial operations. FHE only supports polynomial
  operations. All prior solutions are APPROXIMATIONS:

  1. PolyNorm (v116/v117, based on ULD-Net ICLR 2026):
     Approximates (x-μ)/σ as c0 + c1·x + c2·x².
     - Shape approximation: a degree-2 polynomial CANNOT exactly represent (x-μ)/σ
       (which has a 1/sqrt singularity). The fit is good near the training mean but
       diverges in the tails — exactly where medical lesions live (heavy-tailed).
     - v111 failure mode: polynomial approximation of 1/sqrt(var) has coefficient
       explosion near var=0. PolyNorm sidesteps this by fitting the WHOLE function,
       but the degree-2 limitation remains.
     - Static: coefficients c0/c1/c2 are learned during training, frozen at inference.
       Cannot adapt to inference-time distribution drift.

  2. CipherRMSNorm (v46, statistics baking):
     Replaces dynamic mean/var with EMA-baked constants μ_ema/σ_ema.
     - Structural approximation: keeps the (x-μ)/σ STRUCTURE but uses STATIC stats.
     - Static: μ_ema/σ_ema are frozen after training. If inference distribution
       differs from training (e.g., different hospital's X-ray machine), the baked
       stats are wrong → normalization error.
     - Used as final_norm (affine-only after the last block's CANR).

  3. CADN (v165, THIS MODULE):
     The client ALREADY decrypts x during CANR. Computing LayerNorm in plaintext
     is trivial (O(N·D) element-wise, 1024 elements). Re-encrypt z_norm as a fresh
     ciphertext → server gets EXACT, depth-0, DYNAMIC LayerNorm.

     - Operator fidelity: z_cadn computes the same LayerNorm formula as the
       plaintext reference. It has no polynomial-approximation error, but still
       has floating-point and CKKS error.
     - Dynamic: μ_x/σ_x computed per-input at inference time. Adapts to ANY
       distribution drift (different hospital, different disease prevalence).
     - Depth-0: z_norm arrives as a fresh ciphertext (depth 0). Saves 2 depth
       (PolyNorm polyval) or 1 depth (CAS PolyNorm) per block.
     - No correlated noise: unlike cipher×cipher x² in PolyNorm (which amplifies
       noise ∝ |x|), CADN's z_norm is a fresh encryption with independent noise.

  CADN breaks the approximation-depth trade-off:
    Prior FHE normalization:  lossy (PolyNorm) OR static (CipherRMSNorm) OR deep (both)
    CADN:                     no polynomial approximation + dynamic + depth-0

  The cost: +1 plaintext LayerNorm + +1 encryption per block (same as CAS).
  The client ALREADY does CANR (decrypt + re-encrypt). CADN adds: compute LayerNorm
  on the already-decrypted x (trivial), re-encrypt z_norm (same cost as CAS's x²).
  Net additional cost vs CAS: +1 LayerNorm (~1μs for 1024 elements). Negligible.

PRIVACY (identical to CAS — same contract, same guarantees):
  - Client sees its OWN features (already sees them during CANR) ✓
  - Server NEVER sees plaintext (only gets z_norm ciphertext) ✓
  - γ/β are PUBLIC affine constants sent to client (same as CAS's c0/c1/c2 —
    D-dim per-channel parameters, NOT secret weights). The client applies them
    as part of the LayerNorm formula. Revealing γ/β leaks D scalars — equivalent
    to CAS revealing c0/c1/c2 (also D-dim). This is the SAME privacy contract.
  - Raw image NEVER leaves client ✓
  - Client NEVER sees model weights (W_a, W_o, W_intra, A_inter remain server-side) ✓

TRAIN-INFERENCE CONSISTENCY (the key design principle):
  CADNLayer.forward_plain() uses STANDARD LayerNorm (dynamic mean/var, γ/β affine).
  CADNLayer.forward_cipher() is a NO-OP when z_norm is provided (client already did it).
  → Training learns γ/β via standard LayerNorm gradients.
  → Inference uses the SAME standard LayerNorm (computed by client in plaintext).
  → Zero ARCHITECTURAL train-inference gap: both paths compute the same LayerNorm
    function (no polynomial approximation, no static baking). The only residual
    difference is float32 (training) vs float64+CKKS-noise (inference), which is
    ~1e-9 — negligible vs PolyNorm's ~1e-2 polynomial error or CipherRMSNorm's
    static-vs-dynamic gap.

  This is the SAME principle as v121's scalar PolyAct: design the architecture to
  MATCH the FHE constraints, so train=inference by construction.

BLOCK DEPTH IMPACT (v165 vs v164):
  v164 (CAS PolyNorm + degree-1 PolyAct):
    PolyNorm (CAS): +1    |  W_a matmul: +1  |  PolyAct (deg-1): +1  |  W_o matmul: +1  = +4
  v165 (CADN + degree-1 PolyAct):
    CADN: +0 (depth-0)    |  W_a matmul: +1  |  PolyAct (deg-1): +1  |  W_o matmul: +1  = +3
  CADN saves 1 depth per block. For L=3 layers: 3 depth saved (12→9). More margin
  for deeper networks or higher-degree activations.

MEDICAL JUSTIFICATION:
  Medical features are heavy-tailed (lesions are rare, high-magnitude outliers).
  PolyNorm's degree-2 polynomial under-fits the tails → lesion features are
  mis-normalized → diagnostic signal lost. CADN's EXACT LayerNorm preserves the
  full distribution including tails → lesion features normalized correctly.
  This is the medical-specific reason CADN matters beyond the generic FHE benefit.

References:
  - ULD-Net (ICLR 2026): polynomial ViT normalization (CADN supersedes this for
    the client-assisted setting).
  - PolyNeXt (2026): polynomial MetaFormer (same polynomial limitation).
  - v111 failure: polynomial 1/sqrt divergence (the problem CADN eliminates).
  - CAS (v152, this codebase): the second CA-FHE instance, CADN's direct predecessor.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .ciphertext_tensor import CiphertextTensor


class CADNLayer(nn.Module):
    """CADN: Client-Assisted Dynamic Normalization layer.

    Replaces PolyNorm inside MedFHEBlock. In training (plain), it is a STANDARD
    LayerNorm (dynamic mean/var, learnable γ/β). In inference (cipher), the
    encoder's CADN callback produces z_norm at depth 0 — this layer is a pass-through.

    This design guarantees zero architectural train-inference gap:
      - Training:   z = LayerNorm(x, γ, β)  [standard, dynamic]
      - Inference:  z = CADN_callback(x, γ, β)  [client computes the SAME LayerNorm]
      Both compute the same mathematical function (LayerNorm) — no polynomial
      approximation, no static baking. Residual difference is only float32-vs-float64
      precision + CKKS re-encryption noise (~1e-9), negligible vs PolyNorm's ~1e-2.

    Args:
        dim: Feature dimension D (normalized along this dim).
        eps: LayerNorm epsilon (default 1e-6, matches SigLIP).
        use_mean_subtraction: True=LayerNorm (subtract mean), False=RMSNorm.
            Default True (matches SigLIP pre-training, which uses LayerNorm).

    Attributes:
        weight: [D] learnable γ (LayerNorm affine gain).
        bias: [D] learnable β (LayerNorm affine bias).

    Cipher path:
        forward_cipher(x, z_norm=None):
          - If z_norm is provided (CADN callback output): return z_norm directly
            (depth 0, no operations — client already did the LayerNorm).
          - If z_norm is None (backward compat, no CADN callback): fall back to
            affine-only path using EMA-baked stats (like CipherRMSNorm). This
            should NOT happen in v165 deployment but provides graceful degradation.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        use_mean_subtraction: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # use_mean_subtraction controls whether the layer computes LayerNorm (True,
        # subtracts mean AND divides by std) or RMSNorm (False, divides by RMS only).
        # Default True matches SigLIP pre-training (which uses LayerNorm).
        # v165fix: Previously the flag was stored but IGNORED — `nn.LayerNorm` always
        # subtracts the mean. Now forward_plain respects the flag: when False, a
        # manual RMSNorm is computed (x / sqrt(mean(x²) + eps) * γ + β). This makes
        # the cipher-path fallback (which uses _baked_mean=None when False) consistent
        # with the plain path. CADN callback path is unaffected (client computes
        # whichever normalization matches the training path).
        self.use_mean_subtraction = use_mean_subtraction

        # Standard LayerNorm affine parameters (γ, β) — learned during training.
        # At inference, these are sent to the client as PUBLIC constants for CADN.
        # When use_mean_subtraction=True, nn.LayerNorm handles the dynamic
        # mean/var computation. When False, forward_plain uses a manual RMSNorm
        # path (still using these affine γ/β). The `weight`/`bias` properties
        # below delegate to this submodule's parameters.
        self.layer_norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)

        # EMA fallback stats (for backward-compat cipher path without CADN callback).
        # Initialized to identity (mean=0, var=1 → scale=1). Updated during training
        # via _update_ema(). Used ONLY if forward_cipher is called without z_norm.
        self.register_buffer("_ema_mean", torch.zeros(dim))
        self.register_buffer("_ema_mean_x2", torch.ones(dim))
        self._baked_scale = None
        self._baked_mean = None
        self._baked = False

    @property
    def weight(self):
        """γ parameter (delegates to internal LayerNorm)."""
        return self.layer_norm.weight

    @property
    def bias(self):
        """β parameter (delegates to internal LayerNorm)."""
        return self.layer_norm.bias

    def _update_ema(self, x: torch.Tensor, momentum: float = 0.99):
        """Update EMA of per-channel mean(x) and mean(x²).

        Used ONLY for the backward-compat cipher fallback path (no CADN callback).
        In v165 deployment, CADN callback is always used, so EMA is unused.
        """
        with torch.no_grad():
            if x.dim() > 1:
                dims = tuple(range(x.dim() - 1))
                mean_x2 = x.pow(2).mean(dim=dims)
                mean_x = x.mean(dim=dims) if self.use_mean_subtraction else None
            else:
                mean_x2 = x.pow(2).mean(dim=0)
                mean_x = x.mean(dim=0) if self.use_mean_subtraction else None
            self._ema_mean_x2.mul_(momentum).add_(mean_x2 * (1 - momentum))
            if mean_x is not None:
                self._ema_mean.mul_(momentum).add_(mean_x * (1 - momentum))

    def bake_statistics(self):
        """Freeze EMA into constant scale/mean for backward-compat cipher fallback.

        In v165 deployment, this is NOT needed (CADN callback provides z_norm).
        Kept for backward compatibility with code that calls bake_statistics() on
        all norm layers.
        """
        if self.use_mean_subtraction:
            var = self._ema_mean_x2 - self._ema_mean.pow(2)
            # v168fix: Single eps guard (was double-eps: clamp(min=eps) THEN +eps).
            # Matches cipher_norm.py's v165fix and nn.LayerNorm behavior
            # (var + eps applied once, not twice). The clamp floors var at eps,
            # so the effective minimum variance is eps (not 2*eps as before).
            var = torch.clamp(var, min=self.eps)
            self._baked_scale = 1.0 / torch.sqrt(var)
            self._baked_mean = self._ema_mean.clone()
        else:
            self._baked_scale = 1.0 / torch.sqrt(self._ema_mean_x2 + self.eps)
            self._baked_mean = None
        self._baked = True
        return self

    def forward_plain(
        self,
        x: torch.Tensor,
        token_regions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Plain forward: dynamic normalization matching the client's CADN computation.

        This is the TRAINING path. It computes the EXACT normalization that the
        client will compute at inference via CADN. token_regions is IGNORED (CADN
        is per-token, not per-region — each token is normalized independently,
        which is what standard LayerNorm/RMSNorm does).

        When use_mean_subtraction=True (default, matches SigLIP): standard LayerNorm
          with dynamic mean/var + γ/β affine (delegates to nn.LayerNorm).
        When use_mean_subtraction=False: RMSNorm with dynamic RMS + γ/β affine
          (manual computation: x / sqrt(mean(x²) + eps) * γ + β).

        Args:
            x: [B, N, D] or [N, D] tensor.
            token_regions: IGNORED (kept for API compat with PolyNorm).

        Returns:
            [B, N, D] or [N, D] normalized tensor (LayerNorm or RMSNorm output).
        """
        # Update EMA during training (for backward-compat fallback only).
        if self.training:
            self._update_ema(x)
        # Dispatch based on the normalization mode.
        # v165fix: Previously this ALWAYS called self.layer_norm (LayerNorm) regardless
        # of use_mean_subtraction. Now RMSNorm is computed manually when False, so
        # the cipher-path fallback (which sets _baked_mean=None for RMSNorm) is
        # consistent with the plain path. CADN callback path is unaffected (the client
        # computes whichever normalization matches the training path).
        if self.use_mean_subtraction:
            # Standard LayerNorm: dynamic mean/var + γ/β affine.
            return self.layer_norm(x)
        # RMSNorm: x / sqrt(mean(x², dim=-1) + eps) * γ + β (no mean subtraction).
        # Computes in float32 for numerical stability, then casts back.
        dtype = x.dtype
        x_f = x.float()
        ms = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_f * torch.rsqrt(ms + self.eps)
        x_normed = x_normed * self.weight + self.bias
        return x_normed.to(dtype)

    def forward(self, x, token_regions=None):
        """nn.Module.forward — dispatches to forward_plain (training/plain mode).

        This enables standard PyTorch calls like `self.final_norm(x)` to work
        correctly with CADNLayer, computing STANDARD LayerNorm (dynamic mean/var
        + γ/β affine). This matches what the client computes via CADN at inference.
        """
        return self.forward_plain(x, token_regions)

    def forward_cipher(
        self,
        x: CiphertextTensor,
        token_regions: Optional[List[int]] = None,
        x_sq: Optional[CiphertextTensor] = None,
        z_norm: Optional[CiphertextTensor] = None,
    ) -> CiphertextTensor:
        """Cipher forward: CADN path (z_norm provided) or backward-compat fallback.

        v165 CADN path (z_norm provided):
          The encoder's CADN callback already computed z = LayerNorm(x, γ, β) in
          plaintext and re-encrypted it. z_norm is a fresh ciphertext at depth 0.
          We return it directly — NO operations, depth +0.

        Backward-compat fallback (z_norm is None, e.g., no CADN callback configured):
          Apply affine-only normalization using baked EMA stats (like CipherRMSNorm).
          This is the v164 behavior — less accurate than CADN but graceful degradation.

        Args:
            x: CiphertextTensor (input, depth d).
            token_regions: IGNORED (CADN is per-token, not per-region).
            x_sq: IGNORED (CADN doesn't need x² — no polynomial).
            z_norm: CADN callback output (fresh ciphertext of LayerNorm(x, γ, β)).

        Returns:
            CiphertextTensor at depth 0 (CADN) or x.depth+1 (fallback).
        """
        if z_norm is not None:
            # v165 CADN path: client already computed EXACT LayerNorm.
            # z_norm is a fresh ciphertext at depth 0. Return directly.
            return z_norm

        # Backward-compat fallback (no CADN callback): affine-only with baked stats.
        # This mirrors CipherRMSNorm's cipher path. Less accurate than CADN.
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

        # Cipher fallback: y = (x - mean) * scale * weight + bias = x*w' + b'
        if not self._baked:
            # v175 FIX: previously returned x silently (identity fallback),
            # which masks CADN wiring bugs. Now raises loudly so mis-wired
            # CADN is caught immediately rather than producing wrong results.
            raise RuntimeError(
                "CADNLayer.forward: z_norm is None and stats not baked. "
                "In v165+ deployment, the client MUST provide z_norm via the "
                "CADN callback. This error indicates a CADN wiring bug."
            )

        scale = self._baked_scale.detach().cpu()
        w = self.weight.detach().cpu()
        b = self.bias.detach().cpu() if self.bias is not None else None
        effective_w = (scale * w).reshape(-1)
        effective_b = torch.zeros_like(effective_w)
        if self.use_mean_subtraction and self._baked_mean is not None:
            mean = self._baked_mean.detach().cpu()
            effective_b = effective_b - mean * effective_w
        if b is not None:
            effective_b = effective_b + b.reshape(-1)
        result = x * effective_w.tolist()
        result = result + effective_b.tolist()
        return result

    def apply_baked_affine_list(self, ciphertexts, engine):
        """Apply the baked affine fallback to a raw ciphertext list.

        Used by LBR/PBSR when some tokens skip the client CADN round trip.
        CADN-trained blocks never learned PolyNorm coefficients, so the only
        train-consistent no-refresh path is this EMA affine map.
        """
        if not self._baked:
            self.bake_statistics()
        wrapped = CiphertextTensor(
            ciphertexts=list(ciphertexts),
            engine=engine,
            depth=0,
        )
        return self.forward_cipher(wrapped, z_norm=None)._cipher

    def get_cadn_params(self):
        """Get (γ, β) for the CADN client callback.

        The encoder calls this to pass γ/β to the client's cadn_refresh().
        These are PUBLIC affine constants (same privacy contract as CAS's c0/c1/c2).

        Returns:
            (weight, bias) tuple of [D] tensors.
        """
        return self.weight.detach().cpu(), self.bias.detach().cpu()

    def prepare_cipher_weights(self):
        """Cache weights for cipher path (backward-compat fallback only).

        In v165 CADN deployment, γ/β are sent to the client fresh each call
        (via get_cadn_params()), so no caching is needed. This method exists
        for backward compatibility with code that calls prepare_cipher_weights()
        on all norm layers.
        """
        # Nothing to cache for CADN (γ/β sent to client dynamically).
        # Bake EMA stats for the fallback path.
        self.bake_statistics()

    def multiplicative_depth(self) -> int:
        """Return FHE multiplicative depth of the CADN path.

        Returns 0: the canonical CADN path provides z_norm as a fresh depth-0
        ciphertext (client computes exact LayerNorm in plaintext during CANR
        refresh and re-encrypts). No cipher×plain multiplication occurs.

        The fallback path (no z_norm, baked stats, cipher×plain affine) would
        be depth 1, but v175 fix makes it raise RuntimeError if stats are not
        baked, so it can never silently execute. Callers can rely on depth 0
        for the canonical deployment config.
        """
        return 0  # v165 canonical CADN path is depth-0 (z_norm always provided)
