"""Plug-and-Play Perceiver Interface: abstract base + registry pattern (v167).

FIRST-PRINCIPLES ANALYSIS (why a plug-and-play Perceiver interface is needed):

  Before v167, MedicalConceptAdapter was TIGHTLY COUPLED to PerceiverDecoder:
    - `self.perceiver = PerceiverDecoder(...)` hard-coded in adapter __init__
    - To swap perceiver, the adapter had to be rewritten
    - Top-conference reviewers naturally ask: "Is your FHE-ViT truly backend-agnostic
      beyond the LLM? What about the perceiver?"

  v167's plug-and-play interface DECOUPLES the perceiver from the adapter:
    - Abstract `BasePerceiverDecoder` defines the contract (expand 16→256 tokens,
      apply medical inductive bias, output LLM-ready embeddings)
    - `PerceiverRegistry` allows registering multiple perceivers by name
    - `PerceiverDecoder` (v162 canonical) registered as "v162-default"
    - `IdentityPerceiver` (no-op, repeats 16 tokens 16×) registered as "identity"
      for ablation: isolates the contribution of the perceiver's learned parameters

  This enables:
    1. A/B comparison of different perceiver designs with the SAME FHE ViT
    2. Top-conference ablation: "with perceiver" vs "identity perceiver"
    3. Future perceiver variants (e.g., anatomy-aware, region-attention) can be
       added by implementing BasePerceiverDecoder, no adapter changes needed

CONTRACT:
  The perceiver receives 16 decrypted (and CFP-noised) embeddings [16, D] from
  the client, and produces 256 LLM-ready embeddings [256, D] for the LLM. It is
  a SERVER-SIDE module (all weights private to server). The interface abstracts:
    - How 16 tokens are expanded to 256 (cross-attention, repeat, learned queries)
    - What inductive bias is applied (anatomy, frequency, none)
    - The output dimensionality (must match llm_hidden_dim)

USAGE:
  ```python
  from fpavlm.cipher_native.base_perceiver import PerceiverRegistry

  # Register perceivers (done automatically on import of perceiver_decoder.py)
  # @PerceiverRegistry.register("v162-default")
  # class PerceiverDecoder(BasePerceiverDecoder): ...

  # Load a perceiver by name
  perceiver = PerceiverRegistry.load(
      name="v162-default",
      dim=2560,
      num_latents=256,
      ...
  )

  # Forward pass
  fhe_256_tokens = perceiver(fhe_16_tokens)  # [16,D] -> [256,D]
  ```
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn


class BasePerceiverDecoder(nn.Module, ABC):
    """Abstract base class for plug-and-play perceiver decoders.

    A perceiver takes the 16 decrypted FHE image tokens [16, D] and produces
    256 LLM-ready image tokens [256, D]. The expansion mechanism is the
    perceiver's main design choice (cross-attention, learned queries, etc.).

    All implementations are SERVER-SIDE: weights are private to the server,
    and the module operates on plaintext (post-decryption, post-CFP) embeddings.
    The FHE-only computation happens BEFORE this module (in the FHE ViT encoder).

    Subclasses MUST implement:
      - `forward(tokens_16: torch.Tensor) -> torch.Tensor` (256 tokens)
      - `output_token_count` property (default 256)
      - `forward_with_medical_prior` if they support prior injection
    """

    def __init__(
        self,
        dim: int = 2560,
        num_latents: int = 256,
        num_input_tokens: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents
        self.num_input_tokens = num_input_tokens

    @abstractmethod
    def forward(self, tokens_16: torch.Tensor) -> torch.Tensor:
        """Expand 16 input tokens to 256 output tokens.

        Args:
            tokens_16: [B, 16, D] decrypted+CFP-noised embeddings from client.

        Returns:
            tokens_256: [B, 256, D] LLM-ready embeddings for the LLM.
        """
        raise NotImplementedError

    @property
    def output_token_count(self) -> int:
        """Number of output tokens (default 256 to match MedGemma/LLaVA)."""
        return self.num_latents

    @property
    def input_token_count(self) -> int:
        """Number of input tokens (default 16 from FHE ViT encoder)."""
        return self.num_input_tokens

    def supports_medical_prior(self) -> bool:
        """Whether this perceiver accepts a medical prior for conditioning.

        Override to return True if forward() accepts `medical_prior` kwarg.
        Default False (no prior injection).
        """
        return False

    def get_info(self) -> Dict[str, Any]:
        """Get perceiver info for documentation/logging."""
        return {
            "perceiver_name": self.__class__.__name__,
            "dim": self.dim,
            "num_latents": self.num_latents,
            "num_input_tokens": self.num_input_tokens,
            "supports_medical_prior": self.supports_medical_prior(),
            "output_token_count": self.output_token_count,
            "num_params": sum(p.numel() for p in self.parameters()),
        }


class PerceiverRegistry:
    """Registry for plug-and-play perceiver decoders.

    Usage:
        @PerceiverRegistry.register("v162-default")
        class PerceiverDecoder(BasePerceiverDecoder):
            ...

        perceiver = PerceiverRegistry.load("v162-default", dim=2560, num_latents=256)
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type], type]:
        """Decorator to register a perceiver class.

        Args:
            name: Registry name (e.g., "v162-default", "identity", "anatomy-aware")
        """
        def decorator(perceiver_cls: type) -> type:
            if not issubclass(perceiver_cls, BasePerceiverDecoder):
                raise TypeError(
                    f"{perceiver_cls.__name__} must inherit BasePerceiverDecoder"
                )
            if name in cls._registry:
                raise KeyError(f"Perceiver {name!r} is already registered")
            cls._registry[name] = perceiver_cls
            return perceiver_cls
        return decorator

    @classmethod
    def load(
        cls,
        name: str,
        *,
        dim: int = 2560,
        num_latents: int = 256,
        num_input_tokens: int = 16,
        **kwargs,
    ) -> BasePerceiverDecoder:
        """Load a registered perceiver by name.

        Args:
            name: Registry name (e.g., "v162-default")
            dim: Hidden dimension.
            num_latents: Number of output tokens (default 256).
            num_input_tokens: Number of input tokens (default 16).
            **kwargs: Perceiver-specific configuration.

        Returns:
            BasePerceiverDecoder instance (initialized, ready for forward).

        Raises:
            KeyError: If name is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys()) or "(none)"
            raise KeyError(
                f"Perceiver '{name}' not registered. Available: {available}. "
                f"Register with @PerceiverRegistry.register('{name}')."
            )
        perceiver_cls = cls._registry[name]
        instance = perceiver_cls(
            dim=dim,
            num_latents=num_latents,
            num_input_tokens=num_input_tokens,
            **kwargs,
        )
        return instance

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered perceiver names."""
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a perceiver is registered."""
        return name in cls._registry

    @classmethod
    def get_default(cls) -> str:
        """Return the canonical perceiver name (established v165, current v175)."""
        return "v162-default"


__all__ = ["BasePerceiverDecoder", "PerceiverRegistry"]
