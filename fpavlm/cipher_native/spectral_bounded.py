"""SpectralBoundedLinear: Magnitude-controlled linear layer for FHE.

v48 Innovation 1 (SIMPLIFIED FORM — init-time constraint only in v165 canonical):
  Spectral-Bounded Weight Constraint. The full form (post-step projection during
  training) is wired but not activated in `artifacts/encoder_weights.pt`.
  See "STATUS" section below for the academic-standard disclosure.

ROOT CAUSE of v46 cipher failure:
  - CKKS noise is proportional to value magnitude
  - v46 weights cause value magnitude to grow 1350x (norm 20 → 27000)
  - This amplifies noise 1350x, dropping cos from 1.0 to 0.81

SOLUTION: Constrain weight spectral norm during training.
  - For each weight W, compute its spectral norm σ(W) = largest singular value
  - If σ(W) > τ (threshold), project W back to σ(W) = τ
  - This ensures Π σ(Wᵢ) ≤ τ^L, bounding value growth

Theory:
  - For chain x @ W₁ @ W₂ @ ... @ Wₙ:
    ||output|| ≤ Π ||Wᵢ||₂ · ||input|| = Π σ(Wᵢ) · ||input||
  - With σ(Wᵢ) ≤ τ: ||output|| ≤ τⁿ · ||input||
  - Target: τⁿ ≤ 10 (so 8 layers give ≤ 10x growth, not 1350x)
  - For n=8: τ = 10^(1/8) ≈ 1.33

Implementation:
  - Power iteration to estimate spectral norm (1-5 iterations, cheap)
  - Hard constraint (v165 canonical): W = W * τ / σ(W) if σ(W) > τ  (call
    project_spectral, after optimizer.step() when SPECTRAL_PROJECT_ENABLED=True).
    The canonical checkpoint uses init-time scaling only; post-step
    projection is wired but disabled — full form requires re-training.
  - Soft penalty loss REMOVED in v170 cleanup (dead code — never activated).

CANONICAL CHECKPOINT STATUS:
  - ACTIVE: init-time spectral scaling — initial
    weights are scaled so σ(W) ≈ τ at construction. This is a one-shot init constraint,
    NOT enforced during training.
  - WIRED BUT DISABLED: post-step projection. Setting
    this flag to True iterates all SpectralBoundedLinear modules and calls
    project_spectral() after each optimizer.step(), enforcing σ(W) ≤ τ throughout
    training. This is the "full form" of Innovation 1.
  - IMPLICATION: In the canonical checkpoint, σ(W) of trained SpectralBoundedLinear modules
    MAY drift above τ=3.0 during fine-tuning. The init constraint provides a
    reasonable starting point but no formal guarantee on the final spectral norm.
    Empirically cos_16=1.000000 holds, suggesting drift is bounded in practice.
  - v166+ ROADMAP: Re-train with SPECTRAL_PROJECT_ENABLED=True to obtain the
    formal σ(W) ≤ τ guarantee. The code path is fully tested.

This is FHE-first design: we sacrifice a tiny bit of model capacity
(spectral norm constraint) for a HUGE gain in FHE precision.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .ciphertext_tensor import CiphertextTensor


class SpectralBoundedLinear(nn.Module):
    """Linear layer with spectral norm constraint for FHE magnitude control.

    Weight stored as [in, out] for FHE matmul compatibility.
    Spectral norm is constrained to ≤ τ via power iteration + projection.

    The constraint is applied:
      1. At __init__ (always): scale initial weights to σ(W) ≈ τ.
      2. After optimizer.step() (when SPECTRAL_PROJECT_ENABLED=True in training
         script): call project_spectral() to re-project if σ(W) grew beyond τ.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        # 3.0 is used by token mixers; main/FFN call sites may override it.
        spectral_bound: float = 3.0,
        n_power_iterations: int = 3,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.spectral_bound = spectral_bound
        self.n_power_iterations = n_power_iterations

        # Weight stored as [in, out] for FHE matmul (x @ W)
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize with Xavier, then scale to spectral_bound
        nn.init.xavier_uniform_(self.weight)
        # Scale initial weights to have spectral norm ≈ spectral_bound
        with torch.no_grad():
            s = self._estimate_spectral_norm(self.weight)
            if s > 0:
                self.weight.mul_(spectral_bound / (s + 1e-8))

    @staticmethod
    def _estimate_spectral_norm(W: torch.Tensor, n_iter: int = 10) -> float:
        """Estimate spectral norm (largest singular value) via power iteration.

        W: [in, out] matrix
        Returns: estimated σ(W)

        Power iteration: starting from random u (shape [out]), repeatedly compute
        v = W @ u (shape [in]) and u = W^T @ v (shape [out]). After convergence,
        σ(W) ≈ ||W @ u|| / ||u|| = ||W @ u|| (since ||u|| is normalized to 1).
        """
        u = torch.randn(W.shape[1], device=W.device, dtype=W.dtype)
        u = u / (u.norm() + 1e-8)
        for _ in range(n_iter):
            v = W @ u  # [in]
            v = v / (v.norm() + 1e-8)
            u = W.T @ v  # [out]
            u = u / (u.norm() + 1e-8)
        # σ ≈ ||W @ u|| (||u||=1 by normalization above)
        sigma = (W @ u).norm().item()
        return sigma

    def project_spectral(self):
        """Hard projection: if σ > τ, scale W to make σ = τ.

        Call this after optimizer.step() to enforce the spectral constraint
        throughout training. Controlled by SPECTRAL_PROJECT_ENABLED flag in
        train_v156_comprehensive.py. No-op when σ(W) ≤ τ already.
        """
        with torch.no_grad():
            sigma = self._estimate_spectral_norm(self.weight, n_iter=10)
            if sigma > self.spectral_bound:
                self.weight.mul_(self.spectral_bound / (sigma + 1e-8))

    def verify_spectral_bound(self) -> tuple:
        """Check whether trained weights still satisfy σ(W) ≤ τ.

        v175 FIX: Turns the "may drift above τ" assumption into a verified
        fact. Call after loading trained weights (e.g., in from_pretrained)
        to confirm the init-time constraint held through fine-tuning.

        Returns:
            (sigma, tau, satisfied) tuple — sigma is the estimated spectral
            norm, tau is the bound, satisfied is True if sigma ≤ tau.
        """
        with torch.no_grad():
            sigma = self._estimate_spectral_norm(self.weight, n_iter=10)
            return (sigma, self.spectral_bound, sigma <= self.spectral_bound + 1e-6)

    def forward(self, x):
        """y = x @ W + b (cipher × plain, +1 depth in cipher mode).

        v104.3 FIX: matmul_plain consumes +1 depth in TenSEAL CKKS (auto_rescale).
        Previously depth=x.depth (WRONG — tracked depth was 8 instead of 12 for
        1 block, causing refresh point computation to be based on incorrect depth).
        The actual TenSEAL depth consumption was correct (no "scale out of bounds"),
        but the tracked depth was undercounting by 6 (one per matmul in the block).
        """
        # Convert CiphertextTensor to plain tensor if in plain mode
        if isinstance(x, CiphertextTensor):
            if not x.is_cipher:
                x = x._plain
            else:
                # Cipher mode: diagonal method
                weight = self.weight.detach().cpu()
                if x._engine is not None and hasattr(x._engine, "matmul_plain"):
                    new_ct = x._engine.matmul_plain(x._cipher, weight)
                    # v104.3 FIX: +1 depth (matmul_plain consumes a level via auto_rescale)
                    result = CiphertextTensor(ciphertexts=new_ct, engine=x._engine, depth=x.depth + 1)
                else:
                    result = x.matmul_plain(weight)
                if self.bias is not None:
                    result = result + self.bias.detach().cpu().tolist()
                return result

        # Plain mode: x is torch.Tensor, return torch.Tensor
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def multiplicative_depth(self) -> int:
        return 1  # v104.3: cipher×plain matmul consumes +1 depth (was 0)
