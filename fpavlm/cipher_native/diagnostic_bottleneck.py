"""FHE Diagnostic Bottleneck Release (FDBR).

FDBR factorizes an input-space diagnostic projector P = U U^T around the
ciphertext release boundary. For visual tokens x and a private server weight W,

    x P W + b = (x U) (U^T W) + b.

The server evaluates x U under FHE. The client decrypts and perturbs only the
k diagnostic coefficients. The server then applies the private decoder U^T W
before its private Perceiver and LLM. The factorization is exact without noise;
it does not by itself prove that the coefficients contain no private information.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from fpavlm.anatomical_slots import slot_token_groups

from .ciphertext_tensor import CiphertextTensor


@dataclass(frozen=True)
class BottleneckMetadata:
    input_dim: int
    release_dim: int
    output_dim: int
    compression_ratio: float
    decoder_spectral_norm: float
    orthogonality_error: float
    input_token_count: Optional[int] = None
    release_token_count: int = 1
    input_values_per_image: Optional[int] = None
    released_values_per_image: int = 0


class DiagnosticBottleneckRelease:
    """Validated orthogonal diagnostic encoder and private server decoder."""

    def __init__(
        self,
        basis: torch.Tensor,
        output_weight: torch.Tensor,
        output_bias: Optional[torch.Tensor] = None,
        token_groups: Optional[Sequence[Sequence[int]]] = None,
        orthogonality_tolerance: float = 1e-4,
    ):
        basis = torch.as_tensor(basis).detach().cpu().float()
        output_weight = torch.as_tensor(output_weight).detach().cpu().float()
        if basis.ndim != 2 or output_weight.ndim != 2:
            raise ValueError("basis and output_weight must both be rank-2 tensors")
        input_dim, release_dim = basis.shape
        if release_dim < 1 or release_dim > input_dim:
            raise ValueError(
                f"release_dim must be in [1, {input_dim}], got {release_dim}"
            )
        if output_weight.shape[0] != input_dim:
            raise ValueError(
                f"output_weight input dim {output_weight.shape[0]} != basis input dim {input_dim}"
            )
        gram = basis.T @ basis
        identity = torch.eye(release_dim, dtype=basis.dtype)
        orthogonality_error = float(torch.linalg.matrix_norm(gram - identity, ord=2))
        if not torch.isfinite(gram).all() or orthogonality_error > orthogonality_tolerance:
            raise ValueError(
                "diagnostic basis must have orthonormal columns; "
                f"spectral error={orthogonality_error:.3e}"
            )

        output_dim = output_weight.shape[1]
        if output_bias is None:
            output_bias = torch.zeros(output_dim, dtype=output_weight.dtype)
        output_bias = torch.as_tensor(output_bias).detach().cpu().float().reshape(-1)
        if output_bias.numel() != output_dim:
            raise ValueError(
                f"output_bias length {output_bias.numel()} != output dim {output_dim}"
            )

        self.basis = basis.contiguous()
        self.decoder_weight = (basis.T @ output_weight).contiguous()
        self.decoder_bias = output_bias.contiguous()
        self.token_groups = (
            None
            if token_groups is None
            else [list(map(int, group)) for group in token_groups]
        )
        if self.token_groups is not None:
            flattened = [index for group in self.token_groups for index in group]
            if not self.token_groups or any(not group for group in self.token_groups):
                raise ValueError("token groups must be non-empty")
            if len(flattened) != len(set(flattened)) or min(flattened) < 0:
                raise ValueError("token groups must contain unique non-negative indices")
            self.input_token_count = max(flattened) + 1
            if sorted(flattened) != list(range(self.input_token_count)):
                raise ValueError("token groups must partition a contiguous token range")
        else:
            self.input_token_count = None
        decoder_norm = float(torch.linalg.matrix_norm(self.decoder_weight, ord=2))
        if self.token_groups is None:
            input_token_count = None
            release_token_count = 1
            compression_ratio = release_dim / output_dim
            input_values_per_image = None
        else:
            input_token_count = self.input_token_count
            release_token_count = len(self.token_groups)
            input_values_per_image = input_token_count * output_dim
            compression_ratio = (
                release_token_count * release_dim / input_values_per_image
            )
        released_values_per_image = release_token_count * release_dim
        self.metadata = BottleneckMetadata(
            input_dim=input_dim,
            release_dim=release_dim,
            output_dim=output_dim,
            compression_ratio=compression_ratio,
            decoder_spectral_norm=decoder_norm,
            orthogonality_error=orthogonality_error,
            input_token_count=input_token_count,
            release_token_count=release_token_count,
            input_values_per_image=input_values_per_image,
            released_values_per_image=released_values_per_image,
        )

    @property
    def release_dim(self) -> int:
        return self.metadata.release_dim

    @classmethod
    def with_anatomical_slots(
        cls,
        basis: torch.Tensor,
        output_weight: torch.Tensor,
        output_bias: Optional[torch.Tensor] = None,
        *,
        grid_size: int = 4,
        include_background: bool = True,
    ) -> "DiagnosticBottleneckRelease":
        """Construct a release whose tokens are fixed anatomical slots.

        The rank is controlled by ``basis.shape[1]``.  For example, a rank-2
        basis produces 5x2=10 released scalars per image, retaining spatial
        context without exposing the server's LLM hidden representation.
        """
        groups = slot_token_groups(
            num_tokens=grid_size * grid_size,
            grid_size=grid_size,
            include_background=include_background,
        )
        return cls(basis, output_weight, output_bias, token_groups=groups)

    @property
    def release_token_count(self) -> int:
        """Number of released spatial/anatomical slots."""
        return (
            len(self.token_groups)
            if self.token_groups is not None
            else int(self.input_token_count or 0)
        )

    def encode_plain(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.metadata.input_dim:
            raise ValueError(
                f"feature dim {features.shape[-1]} != input dim {self.metadata.input_dim}"
            )
        coefficients = features @ self.basis.to(features.device, features.dtype)
        if self.token_groups is None:
            return coefficients
        if coefficients.shape[-2] != self.input_token_count:
            raise ValueError(
                f"token count {coefficients.shape[-2]} != configured {self.input_token_count}"
            )
        return torch.stack(
            [coefficients[..., group, :].mean(dim=-2) for group in self.token_groups],
            dim=-2,
        )

    def encode_cipher(self, features: CiphertextTensor) -> CiphertextTensor:
        coefficients = features.matmul_plain(self.basis)
        return (
            coefficients
            if self.token_groups is None
            else coefficients.pool_tokens(self.token_groups)
        )

    def decode_plain(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.shape[-1] != self.release_dim:
            raise ValueError(
                f"coefficient dim {coefficients.shape[-1]} != release dim {self.release_dim}"
            )
        weight = self.decoder_weight.to(coefficients.device, coefficients.dtype)
        bias = self.decoder_bias.to(coefficients.device, coefficients.dtype)
        decoded = coefficients @ weight + bias
        if self.token_groups is None:
            return decoded
        if decoded.shape[-2] != len(self.token_groups):
            raise ValueError("pooled coefficient count does not match configured groups")
        expanded = decoded.new_empty(
            (*decoded.shape[:-2], self.input_token_count, decoded.shape[-1])
        )
        for group_index, token_indices in enumerate(self.token_groups):
            expanded[..., token_indices, :] = decoded[..., group_index : group_index + 1, :]
        return expanded

    def projected_reference(self, features: torch.Tensor) -> torch.Tensor:
        """Return the exact projected full-dimensional reference output."""
        return self.decode_plain(self.encode_plain(features))

    def decoded_noise_l2_bound(self, coefficient_noise_l2: float) -> float:
        """Worst-case L2 error after the private linear decoder."""
        if coefficient_noise_l2 < 0:
            raise ValueError("coefficient_noise_l2 must be non-negative")
        return coefficient_noise_l2 * self.metadata.decoder_spectral_norm


__all__ = ["BottleneckMetadata", "DiagnosticBottleneckRelease"]
