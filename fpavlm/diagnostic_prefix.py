"""Backend-specific linear prefix fitted with exact Gaussian noise regularization.

For fixed clipped training releases X and Gaussian noise covariance Sigma,
E|| (X+noise) W - Y ||² adds n*tr(W.T Sigma W) to least squares.
This avoids Monte Carlo training noise and makes the bridge contract explicit.
It does not prove report fidelity or make release noise disappear.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .checkpoint import load_tensor_checkpoint


@dataclass(frozen=True)
class PrefixSpec:
    backend: str
    release_shape: tuple[int, int]
    token_count: int
    hidden_size: int
    encoder_sha256: str
    mechanism_fingerprint: str

    def __post_init__(self):
        if (not isinstance(self.backend, str) or not self.backend
                or not isinstance(self.release_shape, tuple) or len(self.release_shape) != 2
                or any(isinstance(d, bool) or not isinstance(d, int) or d < 1
                       for d in (*self.release_shape, self.token_count, self.hidden_size))):
            raise ValueError("backend and positive tensor dimensions are required")
        for value in (self.encoder_sha256, self.mechanism_fingerprint):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("encoder and release mechanism must be SHA-256 bound")


class DiagnosticPrefixProjector(nn.Module):
    def __init__(self, spec: PrefixSpec):
        super().__init__()
        self.spec = spec
        self.register_buffer("weight", torch.zeros(spec.release_shape[0] * spec.release_shape[1],
                                                   spec.token_count * spec.hidden_size))
        self.register_buffer("bias", torch.zeros(spec.token_count * spec.hidden_size))

    def forward(self, release):
        if release.ndim != 3 or tuple(release.shape[-2:]) != self.spec.release_shape:
            raise ValueError("release shape does not match prefix checkpoint")
        if not torch.isfinite(release).all():
            raise ValueError("release values must be finite")
        value = release.flatten(1).to(self.weight) @ self.weight + self.bias
        return value.reshape(len(release), self.spec.token_count, self.spec.hidden_size)

    @torch.no_grad()
    def fit(self, clipped_release, native_tokens, *, sigma: float, ridge: float = 1.):
        if not math.isfinite(sigma) or sigma < 0 or not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("finite nonnegative sigma and positive ridge are required")
        if (clipped_release.ndim != 3 or tuple(clipped_release.shape[1:]) != self.spec.release_shape
                or tuple(native_tokens.shape) != (len(clipped_release), self.spec.token_count, self.spec.hidden_size)
                or len(clipped_release) < 2):
            raise ValueError("aligned training releases and native tokens must match the prefix contract")
        if not torch.isfinite(clipped_release).all() or not torch.isfinite(native_tokens).all():
            raise ValueError("training tensors must be finite")
        x, y = clipped_release.flatten(1).double(), native_tokens.flatten(1).double()
        x_mean, y_mean = x.mean(0), y.mean(0)
        centered = x - x_mean
        gram = centered.T @ centered + (ridge + len(x) * sigma ** 2) * torch.eye(x.shape[1], device=x.device)
        # Solve on the small input side; large target-token matrices need no inverse.
        weights = torch.linalg.solve(gram, centered.T) @ (y - y_mean)
        self.weight.copy_(weights)
        self.bias.copy_(y_mean - x_mean @ weights)
        return self

    def save(self, path: str, *, training_patient_ids: list[str]):
        torch.save({"state_dict": self.state_dict(), "spec": vars(self.spec),
                    "training_patient_ids": training_patient_ids, "evidence_scope": "development_only"}, path)

    @classmethod
    def load(cls, path: str, *, expected_sha256: str, expected_spec: PrefixSpec):
        value = load_tensor_checkpoint(path, expected_sha256=expected_sha256, required_keys=("spec", "state_dict"))
        actual = PrefixSpec(**value["spec"])
        if actual != expected_spec:
            raise ValueError("prefix encoder, mechanism or backend binding mismatch")
        model = cls(actual)
        model.load_state_dict(value["state_dict"], strict=True)
        if not all(torch.isfinite(t).all() for t in model.buffers()):
            raise ValueError("prefix contains nonfinite parameters")
        return model.eval()
