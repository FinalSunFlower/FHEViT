"""Linear finding probe over a governed FDBR release.

The 1×k diagnostic coefficients are already the client-decrypted sufficient
statistic. This module maps them to the OpenI diagnosis vocabulary so the
server can condition report text on the release itself, without enlarging it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from fpavlm.checkpoint import load_tensor_checkpoint

DEFAULT_CLASSES = (
    "normal",
    "opacity",
    "calcified granuloma",
    "pulmonary atelectasis",
    "cardiomegaly",
)


def fit_ridge(
    features: torch.Tensor,
    labels: torch.Tensor,
    ridge: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (weight [D, C], bias [C], feature_mean [D]) for multi-label ridge."""
    if features.ndim != 2 or labels.ndim != 2:
        raise ValueError("features and labels must be rank-2")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same batch size")
    if features.shape[0] < 2:
        raise ValueError("ridge fit requires at least two rows")
    x = features.detach().float()
    y = labels.detach().float()
    mean = x.mean(dim=0)
    centered = x - mean
    dim = centered.shape[1]
    gram = centered.T @ centered + ridge * torch.eye(dim, dtype=centered.dtype)
    weight = torch.linalg.solve(gram, centered.T @ y)
    bias = y.mean(dim=0) - mean @ weight
    return weight.contiguous(), bias.contiguous(), mean.contiguous()


@dataclass
class FindingProbe:
    weight: torch.Tensor
    bias: torch.Tensor
    feature_mean: torch.Tensor
    classes: tuple[str, ...]
    threshold: float = 0.45
    min_top1: float = 0.35

    def __post_init__(self) -> None:
        if self.weight.ndim != 2 or self.bias.ndim != 1 or self.feature_mean.ndim != 1:
            raise ValueError("probe tensors have incompatible rank")
        if self.weight.shape[1] != self.bias.shape[0] or self.weight.shape[0] != self.feature_mean.shape[0]:
            raise ValueError("probe tensor shapes do not match")
        if len(self.classes) != self.bias.numel():
            raise ValueError("class count does not match probe outputs")
        if not (0.0 <= self.min_top1 <= self.threshold <= 1.0):
            raise ValueError("require 0 <= min_top1 <= threshold <= 1")

    @property
    def input_dim(self) -> int:
        return int(self.weight.shape[0])

    def probabilities(self, coefficients: torch.Tensor) -> torch.Tensor:
        flat = torch.as_tensor(coefficients).detach().float().reshape(-1)
        if flat.numel() != self.input_dim:
            raise ValueError(
                f"release has {flat.numel()} values, probe expects {self.input_dim}"
            )
        logits = (flat - self.feature_mean) @ self.weight + self.bias
        return torch.sigmoid(logits)

    def predicted_labels(self, coefficients: torch.Tensor) -> list[str]:
        probs = self.probabilities(coefficients)
        hits = [
            name
            for name, value in zip(self.classes, probs.tolist())
            if value >= self.threshold
        ]
        if hits:
            return hits
        top_index = int(torch.argmax(probs).item())
        if float(probs[top_index]) >= self.min_top1:
            return [self.classes[top_index]]
        return []

    def describe(self, coefficients: torch.Tensor) -> Optional[str]:
        labels = self.predicted_labels(coefficients)
        if not labels:
            return None
        return "Possible findings from the diagnostic release: " + ", ".join(labels) + "."

    def state_dict(self) -> dict:
        return {
            "weight": self.weight.detach().cpu(),
            "bias": self.bias.detach().cpu(),
            "feature_mean": self.feature_mean.detach().cpu(),
            "classes": list(self.classes),
            "threshold": float(self.threshold),
            "min_top1": float(self.min_top1),
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str,
        expected_sha256: Optional[str] = None,
        required_keys: Sequence[str] = (
            "weight",
            "bias",
            "feature_mean",
            "classes",
        ),
    ) -> "FindingProbe":
        payload = load_tensor_checkpoint(
            path,
            required_keys=tuple(required_keys),
            expected_sha256=expected_sha256,
        )
        return cls(
            weight=torch.as_tensor(payload["weight"]).float(),
            bias=torch.as_tensor(payload["bias"]).float(),
            feature_mean=torch.as_tensor(payload["feature_mean"]).float(),
            classes=tuple(str(name) for name in payload["classes"]),
            threshold=float(payload.get("threshold", 0.45)),
            min_top1=float(payload.get("min_top1", 0.35)),
        )


__all__ = ["DEFAULT_CLASSES", "FindingProbe", "fit_ridge"]
