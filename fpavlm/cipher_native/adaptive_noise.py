"""AdaptiveCKKSNoise: Adaptive CKKS Noise Training (ACNT).

Adaptive CKKS noise training is optional and disabled in the canonical
`artifacts/encoder_weights.pt`. It is fully wired when
`enable_noise_training=True` and `encoder.set_epoch(epoch)` is called.

Canonical status: disabled. The code path remains because it is an explicit,
tested training option rather than a deployment privacy guarantee.

THE TRAIN-INFERENCE GAP (the core problem ACNT addresses):
  - Training: clean torch tensors (zero noise)
  - Inference: real CKKS ciphertexts (multiplicative noise)
  - Model never learns to be robust to actual FHE noise
  - ACNT injects realistic CKKS noise during training to close this gap

FIRST-PRINCIPLES CKKS NOISE MODEL:
  CKKS noise is MULTIPLICATIVE, not additive:
    - Fresh ciphertext: noise sigma_0 ~= 2^-p (p = precision bits, ~40 for TenSEAL)
    - After cipher x plain(m): sigma_new ~= |m| * sigma_old
    - After cipher x cipher: sigma_new ~= sqrt(|c1|^2 sigma_2^2 + |c2|^2 sigma_1^2)
                                  ~= |c| * sigma (symmetric case)
    - After rescaling: sigma_new ~= sigma_old / S (S = scale factor)

  KEY INSIGHT: noise_std proportional to |value| * depth_factor
    depth_factor = (1 + depth_growth)^depth — grows exponentially with depth
    This matches TenSEAL's actual noise growth pattern.

  ACNT injects noise matching this model during training:
    noise = base_rate * (1 + depth_growth)^depth * |x| * epsilon
    where epsilon ~ N(0, 1) is standard Gaussian
    x_noised = x + noise (additive form of multiplicative noise)

  WARMUP: noise is scaled by min(1, epoch / warmup_epochs) to avoid destabilizing
  early training. Full noise is applied after warmup_epochs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveCKKSNoise(nn.Module):
    """Adaptive CKKS Noise Simulator for training-time noise injection.

    Injects multiplicative CKKS-like noise during plaintext training to close
    the train-inference gap. When enabled=False, inject_noise is a no-op.

    Args:
        base_rate: Base noise rate (sigma_0). Default 1e-3.
        depth_growth: Noise growth factor per depth level. Default 0.05.
        warmup_epochs: Number of epochs to ramp up noise from 0 to full. Default 5.
        enabled: If False, inject_noise returns x unchanged. Default True.
    """

    def __init__(
        self,
        base_rate: float = 1e-3,
        depth_growth: float = 0.05,
        warmup_epochs: int = 5,
        enabled: bool = True,
    ):
        super().__init__()
        self.base_rate = base_rate
        self.depth_growth = depth_growth
        self.warmup_epochs = max(warmup_epochs, 1)
        self.enabled = enabled
        self.current_epoch = 0  # Set by encoder.set_epoch()

    def set_epoch(self, epoch: int):
        """Update current epoch for warmup scheduling."""
        self.current_epoch = epoch

    def inject_noise(
        self,
        x: torch.Tensor,
        depth: int = 0,
        is_cipher_x_cipher: bool = False,
    ) -> torch.Tensor:
        """Inject CKKS-like multiplicative noise into tensor.

        noise_std = base_rate * (1 + depth_growth)^depth * warmup_factor * |x|
        x_noised = x + noise_std * epsilon  (epsilon ~ N(0,1))

        For cipher x cipher operations, noise grows as sqrt(2) * |x| (symmetric case).

        When enabled=False, returns x unchanged (no-op).

        Args:
            x: Input tensor [B, N, D] or [B*N, D].
            depth: Current cipher depth (0 = fresh, higher = more noise).
            is_cipher_x_cipher: If True, use cipher×cipher noise model (sqrt(2)×).

        Returns:
            Noised tensor (same shape as x), or x unchanged if disabled.
        """
        if not self.enabled or not self.training:
            return x

        # Warmup factor: ramp from 0 to 1 over warmup_epochs
        warmup_factor = min(1.0, self.current_epoch / self.warmup_epochs)
        if warmup_factor <= 0:
            return x

        # Depth factor: (1 + depth_growth)^depth — exponential growth
        depth_factor = (1.0 + self.depth_growth) ** depth

        # Cipher×cipher multiplier: sqrt(2) for symmetric case
        cipher_multiplier = (2.0 ** 0.5) if is_cipher_x_cipher else 1.0

        # Noise standard deviation: proportional to |x| (multiplicative model)
        noise_std = self.base_rate * depth_factor * warmup_factor * cipher_multiplier

        # Generate noise: Gaussian * |x| (multiplicative noise in additive form)
        x_abs = x.abs()
        gaussian = torch.randn_like(x)
        noise = noise_std * x_abs * gaussian

        return x + noise


__all__ = ["AdaptiveCKKSNoise"]
