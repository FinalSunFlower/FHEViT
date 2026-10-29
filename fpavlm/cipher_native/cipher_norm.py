"""CipherRMSNorm: Bakeable normalization (+1 depth in cipher mode).

v74 Innovation: EMA-based normalization during training (train-inference consistency)

v59 Innovation: Optional mean subtraction (LayerNorm behavior)
v46 Innovation: Statistics Baking for FHE

In plaintext, LayerNorm computes:
  y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

In FHE, the mean/variance computation is IMPOSSIBLE because:
  1. sqrt is non-polynomial
  2. mean requires summing over token dimension (expensive rotations)
  3. division is non-polynomial

Solution (Statistics Baking):
  During training, we track E[mean(x)] and E[mean(x^2)] via EMA.
  At inference time, we REPLACE the dynamic mean/var with EMA values (CONSTANTS).

  LayerNorm: y = (x - EMA_mean) / sqrt(EMA_mean_x2 - EMA_mean^2 + eps) * weight + bias
           = (x - EMA_mean) * scale * weight + bias  (scale is a plain constant)
           = x * scale * weight - EMA_mean * scale * weight + bias
           = x * w' + b'  (affine transform)

  v104.2 DEPTH FIX: Previously documented as "0 depth" — WRONG.
  In TenSEAL CKKS with auto_rescale=True, cipher×plain consumes +1 depth
  (the plain is encoded at the current scale, requiring rescaling).
  So the affine transform x*w' + b' is +1 depth (not 0).

v74 KEY FIX: EMA-based normalization during training
  Problem (v71 analysis): Training/eval used DYNAMIC batch stats for normalization,
  but FHE inference uses BAKED EMA stats. This creates a train-inference gap.
  Additionally, _update_ema was called during eval (bug), contaminating EMA.

  Solution: After a warmup period, use EMA stats (not batch stats) for normalization.
  This ensures training, eval, and FHE inference all use the same normalization.
  Also: only update EMA during training (not eval).

v59: Added use_mean_subtraction option (default True for LayerNorm behavior).
     v45 used LayerNorm (with mean subtraction) and achieved cos=0.91.
     v48-v58 used RMSNorm (no mean subtraction) and plateaued at cos=0.73.
     Mean subtraction prevents feature drift over 12 layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ciphertext_tensor import CiphertextTensor


class CipherRMSNorm(nn.Module):
    """FHE-native normalization with statistics baking.

    Supports both RMSNorm (use_mean_subtraction=False) and LayerNorm (True).
    Training: tracks EMA of per-channel mean(x) and mean(x^2).
    Inference (cipher): uses baked scale and mean (constants), +1 depth.
    Inference (plain): uses same baked values (train-inference consistency).

    v104.2: cipher path is +1 depth (cipher×plain via auto_rescale), not 0.
    """

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True,
                 ema_momentum=0.99, use_mean_subtraction=True,
                 ema_warmup_steps=250):
        """Initialize CipherRMSNorm.

        Args:
            ema_warmup_steps: Number of training steps to use batch stats
                              before switching to EMA-based normalization.
                              During warmup, EMA is updated but not used.
                              After warmup, EMA is used for normalization
                              (train-inference consistency).
                              Default: 250 (≈5 epochs at 50 steps/epoch).
        """
        super().__init__()
        if isinstance(normalized_shape, (list, tuple)):
            self.normalized_shape = tuple(normalized_shape)
        else:
            self.normalized_shape = (int(normalized_shape),)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_mean_subtraction = use_mean_subtraction

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # EMA of per-channel mean(x^2) and mean(x)
        # For normalized_shape=(D,), this is a D-dim vector
        self.register_buffer(
            "_ema_mean_x2",
            torch.ones(self.normalized_shape),
        )
        if use_mean_subtraction:
            self.register_buffer(
                "_ema_mean",
                torch.zeros(self.normalized_shape),
            )
        else:
            self.register_buffer("_ema_mean", None)

        self.ema_momentum = ema_momentum
        self._baked = False
        self._baked_scale = None  # 1 / sqrt(var + eps)
        self._baked_mean = None   # EMA mean (for mean subtraction)

        # v74: EMA-based normalization (train-inference consistency)
        self._ema_warmup_steps = ema_warmup_steps
        self._step_count = 0  # Training step counter (for warmup)

    def _update_ema(self, x: torch.Tensor):
        """Update EMA of mean(x^2) and mean(x) along the normalization dim."""
        with torch.no_grad():
            # Compute mean(x^2) and mean(x) along all dims except the channel dim
            # For ViT: x is [B, N, D], normalize along D (last dim)
            if x.dim() > 1:
                dims = tuple(range(x.dim() - 1))
                mean_x2 = x.pow(2).mean(dim=dims)
                mean_x = x.mean(dim=dims) if self.use_mean_subtraction else None
            else:
                mean_x2 = x.pow(2).mean(dim=0, keepdim=True)
                mean_x = x.mean(dim=0, keepdim=True) if self.use_mean_subtraction else None

            self._ema_mean_x2.mul_(self.ema_momentum).add_(
                mean_x2.flatten()[: self._ema_mean_x2.shape[0]] * (1 - self.ema_momentum)
            )
            if self.use_mean_subtraction and mean_x is not None:
                self._ema_mean.mul_(self.ema_momentum).add_(
                    mean_x.flatten()[: self._ema_mean.shape[0]] * (1 - self.ema_momentum)
                )

    def bake_statistics(self):
        """Freeze the EMA into constant scale and mean (call after training)."""
        if self.use_mean_subtraction:
            # LayerNorm: var = E[x^2] - E[x]^2
            var = self._ema_mean_x2 - self._ema_mean.pow(2)
            # v165fix: Single eps guard (was double-eps: clamp(min=eps) THEN +eps).
            # The old code: var = clamp(var, min=eps); scale = 1/sqrt(var + eps).
            # When var < eps, the clamp floors it at eps, then +eps makes it 2*eps,
            # so the effective minimum variance was 2*eps (not eps). The correct
            # pattern is to clamp at eps (floor) and use that directly (no +eps),
            # which gives effective minimum variance = eps. Both approaches avoid
            # div-by-zero, but the single-eps version is mathematically cleaner
            # and matches nn.LayerNorm's behavior (which uses var + eps once).
            var = torch.clamp(var, min=self.eps)
            self._baked_scale = 1.0 / torch.sqrt(var)
            self._baked_mean = self._ema_mean.clone()
        else:
            # RMSNorm: var = E[x^2] (single +eps, no clamp needed since E[x²] ≥ 0)
            self._baked_scale = 1.0 / torch.sqrt(self._ema_mean_x2 + self.eps)
            self._baked_mean = None
        self._baked = True
        return self

    def forward(self, x):
        """y = (x - mean) * scale * weight + bias (+1 depth in cipher mode).

        v104.2: cipher×plain consumes +1 depth in TenSEAL CKKS (auto_rescale).
        Previously documented as 0 depth — WRONG.
        """

        # Convert CiphertextTensor to plain tensor if in plain mode
        if isinstance(x, CiphertextTensor):
            if not x.is_cipher:
                x = x._plain
            else:
                # Cipher path
                if not self._baked:
                    # Identity (should not happen if bake_statistics was called)
                    return x
                scale = self._baked_scale.detach().cpu()
                w = self.weight.detach().cpu() if self.weight is not None else None
                b = self.bias.detach().cpu() if self.bias is not None else None

                # y = (x - mean) * scale * weight + bias
                # = x * (scale * weight) + (bias - mean * scale * weight)
                # = x * w' + b'  (affine, +1 depth from cipher×plain)
                if w is not None:
                    effective_w = (scale * w).reshape(-1)
                else:
                    effective_w = scale.reshape(-1)

                effective_b = torch.zeros_like(effective_w)
                if self.use_mean_subtraction and self._baked_mean is not None:
                    mean = self._baked_mean.detach().cpu()
                    effective_b = effective_b - mean * effective_w
                if b is not None:
                    effective_b = effective_b + b.reshape(-1)

                # Apply: x * w' + b' (cipher × plain + plain, +1 depth)
                result = x * effective_w.tolist()
                result = result + effective_b.tolist()
                return result

        # Plaintext path (training and plain inference): x is torch.Tensor
        if not self._baked:
            # v74 FIX: Only update EMA during training (not eval)
            # Previously, eval batches contaminated the EMA
            if self.training:
                self._update_ema(x)
                self._step_count += 1

            # v74: After warmup, use EMA stats for normalization
            # This ensures train-inference consistency (FHE uses baked EMA)
            use_ema = self._step_count > self._ema_warmup_steps

            if not use_ema:
                # Warmup phase: use batch stats (dynamic normalization)
                if x.dim() > 1:
                    dims = tuple(range(x.dim() - 1))
                    mean_x2 = x.pow(2).mean(dim=dims, keepdim=True)
                    mean_x = x.mean(dim=dims, keepdim=True) if self.use_mean_subtraction else None
                else:
                    mean_x2 = x.pow(2).mean(dim=0, keepdim=True)
                    mean_x = x.mean(dim=0, keepdim=True) if self.use_mean_subtraction else None
            else:
                # Post-warmup: use EMA stats (train-inference consistency)
                # EMA is detached (no gradient through statistics)
                if x.dim() > 1:
                    shape = [1] * (x.dim() - 1) + [x.shape[-1]]
                else:
                    shape = [1, x.shape[0]]
                mean_x2 = self._ema_mean_x2.detach().view(*shape)
                mean_x = self._ema_mean.detach().view(*shape) if self.use_mean_subtraction else None

            if self.use_mean_subtraction and mean_x is not None:
                # LayerNorm: var = mean(x^2) - mean(x)^2
                # v168fix: Single eps guard (was double-eps: clamp(min=eps) THEN +eps).
                # Matches the v165fix in bake_statistics() above and nn.LayerNorm.
                var = mean_x2 - mean_x.pow(2)
                var = torch.clamp(var, min=self.eps)
                scale = 1.0 / torch.sqrt(var)
                out = (x - mean_x) * scale
            else:
                # RMSNorm: var = mean(x^2)
                scale = 1.0 / torch.sqrt(mean_x2 + self.eps)
                out = x * scale
        else:
            # Baked: use constant scale and mean (train-inference consistency)
            scale = self._baked_scale.to(x.device)
            if x.dim() > 1:
                scale = scale.view(*([1] * (x.dim() - 1)), -1)

            if self.use_mean_subtraction and self._baked_mean is not None:
                mean = self._baked_mean.to(x.device)
                if x.dim() > 1:
                    mean = mean.view(*([1] * (x.dim() - 1)), -1)
                out = (x - mean) * scale
            else:
                out = x * scale

        if self.weight is not None:
            out = out * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def multiplicative_depth(self) -> int:
        return 1  # v104.2: cipher×plain consumes +1 depth (was incorrectly 0)
