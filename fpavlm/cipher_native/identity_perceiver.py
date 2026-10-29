"""IdentityPerceiver: no-op perceiver for ablation and minimal plug-and-play demo.

This perceiver does NOT expand tokens to 256. Instead, it REPEATS the 16 input
tokens 16× to produce 256 tokens (16×16=256). This serves two purposes:

  1. ABLATION BASELINE: Isolates the contribution of the learned PerceiverDecoder.
     If the LLM generates good reports with IdentityPerceiver, the perceiver's
     learned cross-attention is NOT contributing. If the LLM fails (or generates
     degraded reports), the perceiver's learned parameters ARE contributing.

  2. PLUG-AND-PLAY DEMO: Shows that any module implementing BasePerceiverDecoder
     can be swapped in. The framework doesn't depend on PerceiverDecoder's
     internal architecture.

  3. SPEED BASELINE: IdentityPerceiver has 0 parameters → 0 server-side compute.
     Useful for measuring the pure FHE ViT + LLM latency without perceiver overhead.

PRIVACY: UNCHANGED. IdentityPerceiver operates on already-decrypted, CFP-noised
embeddings. It does NOT see raw images. It is server-side but has no learnable
weights, so there is no IP to protect.

NOTE: IdentityPerceiver does NOT use medical_prior (returns False for
supports_medical_prior). This is intentional — to isolate the perceiver's
contribution, we remove ALL prior conditioning.
"""
from __future__ import annotations

from typing import Optional

import torch

from .base_perceiver import BasePerceiverDecoder, PerceiverRegistry


@PerceiverRegistry.register("identity")
class IdentityPerceiver(BasePerceiverDecoder):
    """No-op perceiver: repeat 16 input tokens 16× to produce 256 tokens.

    Used as ablation baseline and minimal plug-and-play demo.

    Output: [B, 256, D] where tokens are: token_0 repeated 16×, then token_1
    repeated 16×, ..., token_15 repeated 16×. This preserves all input
    information without any learned mixing.

    Alternative: tile the 16 tokens to fill 256 positions (token[i % 16]).
    We use REPEAT (not tile) because it groups identical tokens, making it
    easier for the LLM's attention to identify and pool them.
    """

    def __init__(
        self,
        dim: int = 2560,
        num_latents: int = 256,
        num_input_tokens: int = 16,
        repeat_factor: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(dim=dim, num_latents=num_latents, num_input_tokens=num_input_tokens)
        # Compute repeat factor: 256 / 16 = 16 by default
        if repeat_factor is None:
            repeat_factor = num_latents // num_input_tokens
            if repeat_factor * num_input_tokens != num_latents:
                raise ValueError(
                    f"num_latents ({num_latents}) must be divisible by "
                    f"num_input_tokens ({num_input_tokens}) for IdentityPerceiver"
                )
        self.repeat_factor = repeat_factor
        # No parameters — this is a true no-op perceiver

    def forward(self, tokens_16: torch.Tensor) -> torch.Tensor:
        """Repeat 16 input tokens 16× to produce 256 output tokens.

        Args:
            tokens_16: [B, 16, D] decrypted+CFP-noised embeddings.

        Returns:
            tokens_256: [B, 256, D] where each input token is repeated 16×.
        """
        if tokens_16.dim() != 3:
            raise ValueError(f"Expected [B, N, D] tensor, got shape {tokens_16.shape}")
        B, N, D = tokens_16.shape
        if N != self.num_input_tokens:
            raise ValueError(
                f"Expected {self.num_input_tokens} input tokens, got {N}"
            )
        if D != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {D}")
        # Repeat: [B, 16, D] -> [B, 16, 16, D] -> [B, 256, D]
        tokens_256 = tokens_16.unsqueeze(2).expand(-1, -1, self.repeat_factor, -1)
        tokens_256 = tokens_256.reshape(B, N * self.repeat_factor, D)
        return tokens_256

    def supports_medical_prior(self) -> bool:
        return False

    def get_info(self) -> dict:
        info = super().get_info()
        info["repeat_factor"] = self.repeat_factor
        info["num_params"] = 0  # no parameters
        return info


__all__ = ["IdentityPerceiver"]
