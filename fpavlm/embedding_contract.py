"""Typed contracts for the vision-to-language release boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from fpavlm.cipher_native.dp_accountant import DPReport
    from fpavlm.llm.interface import LLMInterface


@dataclass(frozen=True)
class DiagnosticEmbeddingRelease:
    """Client-governed coefficients released after encrypted vision inference."""

    values: torch.Tensor
    dp_report: "DPReport"
    deployment_profile: str
    calibrated: bool

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("diagnostic release must have shape [batch, tokens, rank]")
        if min(self.values.shape) < 1:
            raise ValueError("diagnostic release dimensions must be positive")
        if not torch.isfinite(self.values).all():
            raise ValueError("diagnostic release contains non-finite values")
        if self.dp_report.dim != self.values[0].numel():
            raise ValueError(
                f"DP report dimension {self.dp_report.dim} does not match per-image "
                f"release size {self.values[0].numel()}"
            )
        if self.dp_report.epsilon <= 0 or not 0 < self.dp_report.delta < 1:
            raise ValueError("diagnostic release requires a valid positive DP report")
        if self.deployment_profile != "research_baseline" and not self.calibrated:
            raise ValueError("deployment releases require a feasible held-out calibration")

    @property
    def token_count(self) -> int:
        return self.values.shape[1]

    @property
    def release_dim(self) -> int:
        return self.values.shape[2]

    @property
    def values_per_image(self) -> int:
        return self.token_count * self.release_dim

    @property
    def contains_image_information(self) -> bool:
        """The plaintext boundary is utility-bearing and not zero-leakage."""
        return True

    def composed_epsilon(self, releases: int) -> float:
        """Conservative basic composition across repeated image releases."""
        return self.dp_report.composed_epsilon(releases)


@dataclass(frozen=True)
class LLMVisualEmbedding:
    """Server-private visual tokens immediately before LLM token injection."""

    values: torch.Tensor
    source_release_dim: int
    llm_name: str

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("LLM visual embedding must have shape [batch, tokens, hidden]")
        if min(self.values.shape) < 1 or self.source_release_dim < 1:
            raise ValueError("LLM visual embedding dimensions must be positive")
        if not torch.isfinite(self.values).all():
            raise ValueError("LLM visual embedding contains non-finite values")

    def validate_for(self, llm: "LLMInterface") -> None:
        expected_tokens = llm.expected_image_token_count()
        if self.values.shape[1] != expected_tokens:
            raise ValueError(
                f"{self.llm_name} expects {expected_tokens} image tokens, "
                f"got {self.values.shape[1]}"
            )
        if self.values.shape[2] != llm.llm_hidden_dim:
            raise ValueError(
                f"{self.llm_name} expects hidden dim {llm.llm_hidden_dim}, "
                f"got {self.values.shape[2]}"
            )


__all__ = ["DiagnosticEmbeddingRelease", "LLMVisualEmbedding"]
