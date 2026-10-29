"""Typed interfaces and registry for local Transformers VLM language backends.

Backends inject trained visual prefix embeddings with exact token and hidden
dimensions. Registry compatibility does not establish clinical equivalence or
zero-training transfer. Custom model code requires explicit integrity binding.
"""
from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import torch


class LLMInterface(ABC):
    """Abstract interface for plug-and-play LLM integration.

    Implementations wrap specific LLMs (MedGemma, LLaVA-Med, etc.) to provide
    a uniform interface for the FHE medical VLM pipeline.

    The LLM receives image embeddings from the adapter (MedicalConceptAdapter)
    and generates text (medical report).

    All implementations must be loadable from LOCAL model paths (no HuggingFace
    downloads at runtime — models are pre-downloaded).
    """

    def __init__(
        self,
        model_path: str,
        *,
        allow_remote_code: bool = False,
        remote_code_sha256: Optional[str] = None,
        **kwargs,
    ):
        """Initialize LLM from local model path.

        Args:
            model_path: Local path to the LLM model directory.
            **kwargs: LLM-specific configuration (quantization, device, etc.)
        """
        self.model_path = model_path
        self.allow_remote_code = allow_remote_code
        self.remote_code_sha256 = remote_code_sha256
        self.model = None
        self.processor = None
        self._loaded = False

    def verify_remote_code_trust(self) -> str:
        """Verify the complete local Python source corpus before execution."""
        if not self.allow_remote_code:
            raise PermissionError(
                f"{self.__class__.__name__} requires custom model code. Set "
                "allow_remote_code=True and pin remote_code_sha256 after reviewing "
                "the local model repository."
            )
        expected = (self.remote_code_sha256 or "").lower()
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise ValueError("remote_code_sha256 must be a 64-character SHA-256 digest")
        root = os.path.abspath(os.fspath(self.model_path))
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Model directory not found: {root}")
        sources = []
        for current, directories, files in os.walk(root):
            directories[:] = sorted(
                name for name in directories if name not in {".git", "__pycache__"}
            )
            for filename in sorted(files):
                if filename.endswith(".py"):
                    sources.append(os.path.join(current, filename))
        if not sources:
            raise ValueError(f"No Python source files found under custom model directory {root}")
        digest = hashlib.sha256()
        for source in sorted(sources):
            relative = os.path.relpath(source, root).replace(os.sep, "/")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with open(source, "rb") as handle:
                digest.update(handle.read())
            digest.update(b"\0")
        actual = digest.hexdigest()
        if actual != expected:
            raise PermissionError(
                f"Custom model code digest mismatch: expected {expected}, got {actual}"
            )
        return actual

    @abstractmethod
    def load(self) -> None:
        """Load the LLM model and processor from model_path.

        Must set self.model and self.processor. Called once at initialization.
        For quantized models, configure 4-bit/8-bit loading here.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_image_tokens(
        self,
        image_embeds: torch.Tensor,
        prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """Convert adapter output embeddings to LLM input embedding format.

        Args:
            image_embeds: [B, N_img, llm_hidden_dim] from adapter (plain or
                decrypted FHE output). May be float32 (from FHE decrypt) or
                bfloat16 (from plain path).
            prompt: Optional medical prompt string to prepend (v175: all 4
                concrete adapters accept this kwarg; default None uses the
                adapter's built-in medical prompt).

        Returns:
            inputs_embeds: [1, seq_len, llm_hidden_dim] bfloat16 tensor with
                image embeddings injected at image token positions. Ready for
                model.generate(inputs_embeds=...).
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        image_embeds: torch.Tensor,
        prompt: Optional[str] = None,
        max_new_tokens: int = 200,
        do_sample: bool = False,
        num_beams: int = 1,
        seed: Optional[int] = 42,
        logits_processor: Optional[List] = None,
        num_return_sequences: int = 1,
    ) -> str:
        """Generate text from image embeddings + text prompt.

        Args:
            image_embeds: [B, N_img, llm_hidden_dim] adapter output.
            prompt: Optional text prompt (e.g., "Read this chest X-ray...").
                When None, the adapter uses its built-in medical prompt
                (v175: all 4 concrete adapters accept None and have defaults).
            max_new_tokens: Maximum generation length.
            do_sample: If True, use sampling; if False, greedy/beam.
            num_beams: Number of beams for beam search (1=greedy). Beam search
                is more robust to small input perturbations (e.g., ~1e-5 FHE
                noise) and produces higher-quality reports.
            seed: Random seed for reproducibility (default 42).
            logits_processor: v168 — Optional list of HuggingFace-compatible
                LogitsProcessor objects. When provided, the adapter forwards
                them to `model.generate(logits_processor=...)`. This is the
                integration point for v166 Noise-Aware Decoding mechanisms
                (TopNSigmaLogitProcessor, UncertaintyBoundedLogitProcessor).
                Default None = no processors (standard generation).
            num_return_sequences: v168 — Number of sequences to return per
                prompt. Retained for API compatibility; MBR decoding uses its
                own inline loop over generate() and does not pass this kwarg.
                Default 1.

        Returns:
            Generated text string (medical report). If num_return_sequences>1,
            returns the first sequence of the batch.
        """
        raise NotImplementedError

    @abstractmethod
    def image_token_id(self) -> int:
        """Return the LLM's image token ID (for masked_scatter)."""
        raise NotImplementedError

    @abstractmethod
    def expected_image_token_count(self) -> int:
        """Return the native expected image token count (e.g., 256 for MedGemma).

        The adapter may output fewer tokens (e.g., 16 for MedicalConceptAdapter
        before Perceiver expansion, 256 after).
        The LLM must handle variable image token counts.
        """
        raise NotImplementedError

    def get_native_feature_stats(self) -> Optional[torch.Tensor]:
        """Get native vision feature statistics for adapter calibration.

        v170: Returns the LLM's native vision feature distribution (after the
        vision tower + projector). Used by LLMEmbeddingAdapter to calibrate
        scale/shift so that PCA-adapted FHE features match the native
        distribution.

        Default: returns None (no calibration needed — e.g., MedGemma where
        dim matches and no adapter is used). Override in subclasses that
        support vision feature extraction.

        Returns:
            [B, N_tokens, llm_hidden_dim] native vision features, or None.
        """
        return None

    def extract_native_visual_features(self, image_path: str) -> Optional[torch.Tensor]:
        """Run this LLM's native vision tower on a real image.

        Used by the clinical-equivalence benchmark. Default is None: the
        adapter can generate from injected embeddings but does not expose a
        native-image path.
        """
        return None

    def generate_from_native_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        max_new_tokens: int = 200,
        do_sample: bool = False,
        num_beams: int = 4,
        seed: Optional[int] = 42,
    ) -> str:
        """Generate from the native vision tower, not the FHE student.

        This is the clinical-equivalence upper bound for a given LLM.
        """
        features = self.extract_native_visual_features(image_path)
        if features is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not expose a native vision path. "
                "Clinical equivalence can still compare plaintext-student vs FHE-student."
            )
        if features.dim() == 2:
            features = features.unsqueeze(0)
        return self.generate(
            features,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            seed=seed,
        )

    @property
    def llm_hidden_dim(self) -> int:
        """LLM hidden dimension (e.g., 2560 for MedGemma 4B)."""
        raise NotImplementedError

    @property
    def is_loaded(self) -> bool:
        """Whether the LLM model has been loaded."""
        return self._loaded

    def get_info(self) -> Dict[str, Any]:
        """Get LLM info for documentation/logging."""
        return {
            "llm_name": self.__class__.__name__,
            "model_path": self.model_path,
            "llm_hidden_dim": self.llm_hidden_dim,
            "image_token_id": self.image_token_id() if self._loaded else None,
            "expected_image_token_count": self.expected_image_token_count(),
            "is_loaded": self._loaded,
        }


class LLMRegistry:
    """Registry for plug-and-play LLM adapters.

    Usage:
        @LLMRegistry.register("medgemma-4b")
        class MedGemmaAdapter(LLMInterface):
            ...

        llm = LLMRegistry.load("medgemma-4b", model_path="...")
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type], type]:
        """Decorator to register an LLM adapter class.

        Args:
            name: Registry name (e.g., "medgemma-4b", "llava-med-7b")

        Usage:
            @LLMRegistry.register("medgemma-4b")
            class MedGemmaAdapter(LLMInterface):
                ...
        """
        def decorator(adapter_cls: type) -> type:
            if name in cls._registry:
                raise KeyError(f"LLM adapter {name!r} is already registered")
            cls._registry[name] = adapter_cls
            return adapter_cls
        return decorator

    @classmethod
    def load(cls, name: str, model_path: str, **kwargs) -> LLMInterface:
        """Load a registered LLM by name.

        Args:
            name: Registry name (e.g., "medgemma-4b")
            model_path: Local path to the LLM model directory
            **kwargs: LLM-specific configuration

        Returns:
            LLMInterface instance (loaded and ready for generation)

        Raises:
            KeyError: If name is not registered
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys()) or "(none)"
            raise KeyError(
                f"LLM '{name}' not registered. Available: {available}. "
                f"Register with @LLMRegistry.register('{name}')."
            )
        adapter_cls = cls._registry[name]
        instance = adapter_cls(model_path=model_path, **kwargs)
        instance.load()
        return instance

    @classmethod
    def list_available(cls) -> list:
        """List all registered LLM names."""
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an LLM is registered."""
        return name in cls._registry


__all__ = ["LLMInterface", "LLMRegistry"]
