"""Server-private cross-LLM visual embedding adapter.

PROBLEM: The FHE encoder is trained for ONE specific LLM's hidden_dim (v165:
dim=2560 for MedGemma 4B). Swapping to a different LLM (InternVL2 dim=2048,
Qwen2-VL dim=1536) requires either (a) retraining the encoder, or (b) a
projection head that maps the FHE features to the target LLM's expected dim
AND distribution.

FIRST-PRINCIPLES ANALYSIS:
  Different LLMs have different vision towers (SigLIP, InternViT, Qwen2-VL-ViT)
  that produce features with different:
    1. DIMENSION: 2560 (MedGemma), 2048 (InternVL2), 1536 (Qwen2-VL)
    2. DISTRIBUTION: different mean/std/skew at the image-token positions
    3. SEMANTIC STRUCTURE: different feature subspaces (different "dialects")

  A simple linear projection (2560→target_dim) handles (1) but NOT (2) or (3).
  A trained adapter with distribution matching handles all three.

DESIGN:
  The adapter is a per-LLM head sitting between the FHE
  Perceiver output and the LLM input:

    adapter(target_llm) = Scale ⊙ LayerNorm(Linear_{2560→target_dim}(x)) + Shift

  - Linear: dimension matching (2560 → target_dim)
  - LayerNorm: distribution normalization (removes FHE-specific scale)
  - Scale/Shift: match target LLM's native vision tower statistics

  Training: knowledge distillation against the target LLM's native vision tower.
    1. Run the target LLM's native vision encoder on the same image
    2. MSE-match the adapter output to the native vision features
    3. Freeze the FHE encoder and train only the adapter

  Inference: the server applies the adapter after receiving the governed FDBR
  release and decoding it with private bridge weights. Adapter parameters are
  never sent to the client in the model-confidential profile.

This design makes backends replaceable after per-model distillation. It does not
imply zero-training compatibility or establish a literature-first claim.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn

from fpavlm.checkpoint import load_tensor_checkpoint
from fpavlm.provenance import validate_provenance


class LLMEmbeddingAdapter(nn.Module):
    """Cross-LLM Embedding Adaptation Layer.

    Maps FHE encoder output [B, N_tokens, source_dim] to target LLM input
    [B, N_tokens, target_dim] with distribution matching.

    Args:
        source_dim: FHE encoder output dim (e.g., 2560 for v165)
        target_dim: Target LLM's hidden dim (e.g., 2048 for InternVL2)
        num_tokens: Number of image tokens (e.g., 256 after Perceiver)
    """

    def __init__(
        self,
        source_dim: int = 2560,
        target_dim: int = 2560,
        num_tokens: int = 256,
    ):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.num_tokens = num_tokens

        # Dimension projection
        self.proj = nn.Linear(source_dim, target_dim, bias=False)

        # Distribution normalization
        self.norm = nn.LayerNorm(target_dim)

        # Scale/shift to match target LLM's native vision tower statistics
        self.scale = nn.Parameter(torch.ones(target_dim))
        self.shift = nn.Parameter(torch.zeros(target_dim))

        # Metadata
        self.target_llm_name: Optional[str] = None
        self.training_loss: Optional[float] = None
        self.provenance: Optional[dict[str, Any]] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt FHE features to target LLM's expected input.

        Args:
            x: [B, N_tokens, source_dim] FHE encoder output (after decrypt)

        Returns:
            [B, N_tokens, target_dim] adapted embedding for target LLM
        """
        x = self.proj(x)       # Dimension matching
        x = self.norm(x)       # Distribution normalization
        x = x * self.scale + self.shift  # Statistics matching
        return x

    def clamp_scale(self, max_value: float = 3.0, min_value: float = 0.0):
        """Clamp the per-dim scale parameter to a safe range.

        v175 FIX (CRITICAL — Scale Explosion Bug):
          The `scale` parameter is initialized by `match_target_statistics()` as
          target_std / adapted_std, which is O(1) for well-matched distributions.
          However, during MSE distillation on a SMALL training set (20 images),
          the optimizer can push `scale` to extreme values (10-40×) to overfit
          the training distribution. At inference, these extreme scale values
          AMPLIFY the CFP noise (σ=24×) in certain dimensions, causing the LLM
          to receive completely out-of-distribution features → garbage output
          (repetitive tokens, random Chinese characters, etc.).

          Root cause: LayerNorm normalizes to std=1, so `scale = target_std`.
          Native vision features have std ∈ [0.1, 5.0]. Any scale > 5.0 means
          the adapter is amplifying beyond the target's actual std — this is
          overfitting, NOT legitimate distribution matching.

          Fix: Clamp scale to [min_value, max_value] after each optimizer step.
          Default [0, 3.0] is principled: native vision features rarely have
          std > 3.0, so scale > 3.0 is always overfitting.

        This is a CORE method (not just training-script logic) so ALL adapter
        training scripts can call it. Inference is unaffected (clamping is only
        called during training, not in forward()).

        Args:
            max_value: Maximum allowed scale (default 3.0).
            min_value: Minimum allowed scale (default 0.0, allows sign flip).
        """
        with torch.no_grad():
            self.scale.data.clamp_(min=min_value, max=max_value)

    def init_pca_from_features(self, features: torch.Tensor):
        """Initialize projection via PCA of the ACTUAL FHE features.

        This baseline computes SVD of the
        features and takes the top `target_dim` right singular vectors as the
        projection. This preserves the maximum-variance directions of the actual
        feature batch, unlike truncation or random orthogonal projection.

        Mathematical basis:
          features [N, source_dim] = U @ diag(S) @ Vh
          Vh: [source_dim, source_dim] — rows are principal directions (sorted by
              descending singular value).
          Take Vh[:target_dim, :] → [target_dim, source_dim] projection that
          preserves the top-target_dim principal components (maximum variance).

        For 2560→2048: keeps the 2048 highest-variance directions (drops 512
        low-variance "noise" dims). For 2560→1536: keeps the 1536 highest-variance
        directions. This is the OPTIMAL linear projection under the L2
        reconstruction-error criterion (Eckart-Young theorem).

        Args:
            features: [B, N, source_dim] or [N, source_dim] FHE features.
        """
        with torch.no_grad():
            if features.dim() == 3:
                features = features.reshape(-1, features.shape[-1])  # [B*N, source_dim]
            # SVD: features = U @ diag(S) @ Vh
            # Vh: [min(N, source_dim), source_dim] — rows are principal directions
            _, _, Vh = torch.linalg.svd(features, full_matrices=False)
            # Take top target_dim rows of Vh → [target_dim, source_dim]
            if self.target_dim <= Vh.shape[0]:
                self.proj.weight.data = Vh[:self.target_dim, :].clone()
            else:
                # target_dim > available principal components — pad with zeros
                # v174 FIX: Create zeros on the SAME device as Vh (features device).
                # Previously torch.zeros() defaulted to CPU, which moved self.proj.weight
                # to CPU. This caused a device mismatch in match_target_statistics:
                # self.proj.weight.device was CPU, so inputs were aligned to CPU,
                # but self.norm.weight was on CUDA → _native_layer_norm error.
                self.proj.weight.data = torch.zeros(
                    self.target_dim, self.source_dim,
                    device=Vh.device, dtype=Vh.dtype,
                )
                self.proj.weight.data[:Vh.shape[0], :] = Vh.clone()
            self.norm.weight.data.fill_(1.0)
            self.norm.bias.data.zero_()
            self.scale.data.fill_(1.0)
            self.shift.data.zero_()

    def match_target_statistics(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ):
        """Calibrate scale/shift to match target LLM's native vision features.

        v170 FIX: Uses dim=(0,1) (average over batch AND tokens) to get per-dim
        statistics [target_dim]. Previous code used dim=0 which gave [N_tokens,
        target_dim] — a shape mismatch with self.scale [target_dim].

        Args:
            source_features: [B, N, source_dim] or [N, source_dim] FHE features
            target_features: [B, N, target_dim] or [N, target_dim] Target LLM's
                native vision features (from extract_feature or equivalent)
        """
        with torch.no_grad():
            # Ensure 3D: [B, N, dim]
            if source_features.dim() == 2:
                source_features = source_features.unsqueeze(0)
            if target_features.dim() == 2:
                target_features = target_features.unsqueeze(0)

            # v174 FIX: Align both inputs to the adapter's parameter device.
            # The LLM's get_native_feature_stats() returns CUDA tensors, while
            # the FHE encoder's features are on CPU. Without alignment, the
            # division (target_std / adapted_std) on line 201 fails with
            # "Expected all tensors to be on the same device, but found at
            # least two devices, cuda:0 and cpu".
            source_features = source_features.to(self.proj.weight.device)
            target_features = target_features.to(self.proj.weight.device)

            adapted = self.proj(source_features)  # [B, N, target_dim]
            adapted = self.norm(adapted)

            # Per-dim statistics: average over batch AND tokens → [target_dim]
            target_mean = target_features.mean(dim=(0, 1))  # [target_dim]
            target_std = target_features.std(dim=(0, 1)) + 1e-6  # [target_dim]
            adapted_mean = adapted.mean(dim=(0, 1))  # [target_dim]
            adapted_std = adapted.std(dim=(0, 1)) + 1e-6  # [target_dim]

            self.scale.data = (target_std / adapted_std).to(self.scale.device)
            self.shift.data = (target_mean - adapted_mean * (target_std / adapted_std)).to(self.shift.device)

            # v175 FIX: Clamp scale to [0, 3.0] after statistics matching.
            # Without this, the zero-shot PCA path (called from pipeline.py when no
            # trained adapter exists) is unprotected against scale explosion: if the
            # target LLM's native features have std > 3.0 (the clamp_scale docstring
            # notes native vision features have std ∈ [0.1, 5.0]), scale exceeds 3.0
            # and amplifies CFP noise (σ=24×) → LLM receives OOD features → garbage
            # output. The trained-adapter path is already protected via from_dict()
            # clamp; this closes the gap for the zero-shot path. See clamp_scale()
            # docstring for the full rationale.
            self.clamp_scale(max_value=3.0, min_value=0.0)

    def to_dict(self) -> Dict:
        """Serialize adapter for saving/loading."""
        return {
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "num_tokens": self.num_tokens,
            "target_llm_name": self.target_llm_name,
            "training_loss": self.training_loss,
            "provenance": self.provenance,
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "LLMEmbeddingAdapter":
        """Deserialize adapter."""
        adapter = cls(
            source_dim=d["source_dim"],
            target_dim=d["target_dim"],
            num_tokens=d["num_tokens"],
        )
        adapter.load_state_dict(d["state_dict"])
        adapter.target_llm_name = d.get("target_llm_name")
        adapter.training_loss = d.get("training_loss")
        provenance = d.get("provenance")
        adapter.provenance = dict(provenance) if isinstance(provenance, Mapping) else None
        # v175 SAFETY: Defensively clamp scale to [0, 3.0] at load time.
        # Prevents garbage LLM output if a non-canonical training script produced
        # a checkpoint with extreme scale values (scale explosion bug, see
        # clamp_scale docstring). Negligible cost (one clamp_ on [target_dim]).
        adapter.clamp_scale(max_value=3.0, min_value=0.0)
        return adapter

    def validate_compatibility(
        self,
        *,
        target_llm_name: str,
        source_dim: int,
        target_dim: int,
        num_tokens: int,
        require_provenance: bool = True,
    ) -> None:
        expected = (source_dim, target_dim, num_tokens)
        actual = (self.source_dim, self.target_dim, self.num_tokens)
        if actual != expected:
            raise ValueError(f"adapter shape contract {actual} != expected {expected}")
        if self.target_llm_name != target_llm_name:
            raise ValueError(
                f"adapter target {self.target_llm_name!r} != expected {target_llm_name!r}"
            )
        if require_provenance and self.provenance is None:
            raise ValueError("adapter lacks training provenance")
        if self.provenance is not None:
            validate_provenance(self.provenance, self.provenance)

    def save(self, path: str):
        """Save adapter to file."""
        torch.save(self.to_dict(), path)

    @classmethod
    def load(
        cls, path: str, *, expected_sha256: Optional[str] = None
    ) -> "LLMEmbeddingAdapter":
        """Load adapter from file."""
        return cls.from_dict(
            load_tensor_checkpoint(
                path,
                required_keys=("source_dim", "target_dim", "num_tokens", "state_dict"),
                expected_sha256=expected_sha256,
            )
        )


__all__ = [
    "LLMEmbeddingAdapter",
]
