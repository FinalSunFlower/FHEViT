"""A spatial diagnostic release network with one nonlinear encrypted stage.

Five public CXR slots pool raw patches; a shared learned stem extracts local
features. A learned contralateral difference is added before the square. The
private readout emits k coefficients per slot. Unlike an all-affine student,
this circuit is nonlinear while requiring only three rescale levels.

Fixed slots are anatomical priors, not segmentations or semantic labels.
This architecture is a research candidate; training does not promote privacy
or establish clinical non-inferiority.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from .affine_release import CompiledAffineRelease
from .anatomical_slots import slot_token_groups


@dataclass(frozen=True)
class CompiledSpatialQuadratic:
    stem: CompiledAffineRelease
    linear_weight: torch.Tensor
    square_weight: torch.Tensor
    bias: torch.Tensor

    @property
    def output_shape(self):
        return tuple(self.bias.shape)

    @property
    def multiplicative_depth(self):
        return 3

    def evaluate(self, ciphertexts: list, *, chunk_size: int) -> list:
        hidden = self.stem.evaluate(ciphertexts, chunk_size=chunk_size)
        slots, width, rank = self.linear_weight.shape
        # Balanced product schedule: stem (1), square (2), readout (3).
        squared = [value.square() for value in hidden]
        result = []
        for slot in range(slots):
            for channel in range(rank):
                terms = []
                for j in range(width):
                    i = slot * width + j
                    a = float(self.linear_weight[slot, j, channel])
                    b = float(self.square_weight[slot, j, channel])
                    if a:
                        terms.append(hidden[i] * a)
                    if b:
                        terms.append(squared[i] * b)
                if not terms:
                    import tenseal as ts

                    total = ts.ckks_vector(ciphertexts[0].context(), [0.0])
                else:
                    total = terms[0]
                    for term in terms[1:]:
                        total = total + term
                result.append(total + float(self.bias[slot, channel]))
        return result


class SpatialQuadraticRelease(nn.Module):
    """Trainable raw-patch -> [batch, five slots, rank] medical circuit."""

    def __init__(self, patch_dim: int = 9408, latent_dim: int = 8, rank: int = 4):
        super().__init__()
        if min(patch_dim, latent_dim, rank) < 1:
            raise ValueError("dimensions must be positive")
        self.patch_dim, self.latent_dim, self.rank = patch_dim, latent_dim, rank
        groups = slot_token_groups(num_tokens=16, grid_size=4)
        pooling = torch.zeros(5, 16)
        for slot, group in enumerate(groups):
            pooling[slot, group] = 1 / len(group)
        self.register_buffer("pooling", pooling)
        self.stem_weight = nn.Parameter(torch.randn(patch_dim, latent_dim) / patch_dim ** .5)
        self.stem_bias = nn.Parameter(torch.zeros(5, latent_dim))
        # Only lung slots 0 and 1 are paired; no learned/image-dependent router.
        self.symmetry_gain = nn.Parameter(torch.tensor(.1))
        self.linear_weight = nn.Parameter(torch.randn(5, latent_dim, rank) / latent_dim ** .5)
        self.square_weight = nn.Parameter(torch.randn(5, latent_dim, rank) * .01)
        self.bias = nn.Parameter(torch.zeros(5, rank))

    def stem(self, patches):
        if tuple(patches.shape[-2:]) != (16, self.patch_dim):
            raise ValueError("expected sixteen raw patches with the configured feature dimension")
        pooled = torch.einsum("st,btd->bsd", self.pooling, patches)
        local = pooled @ self.stem_weight + self.stem_bias
        difference = local[:, 0] - local[:, 1]
        correction = torch.stack([difference, -difference, torch.zeros_like(difference),
                                  torch.zeros_like(difference), torch.zeros_like(difference)], dim=1)
        return local + self.symmetry_gain * correction

    def forward(self, patches):
        hidden = self.stem(patches)
        return (torch.einsum("bsl,slk->bsk", hidden, self.linear_weight)
                + torch.einsum("bsl,slk->bsk", hidden.square(), self.square_weight) + self.bias)

    @torch.no_grad()
    def initialize_from_diagnostic_targets(self, patches, targets, *, ridge: float = 1.0):
        """Training-only supervised stem and ridge readout initialization.

        A shared stem comes from pooled-patch / slot-target cross covariance.
        Centering each slot prevents its average brightness from dominating
        the diagnostic directions. The readout includes an explicit intercept.
        This method must never see validation or confirmation patients.
        """
        if patches.ndim != 3 or targets.shape != (len(patches), 5, self.rank):
            raise ValueError("aligned training patches and spatial diagnostic targets are required")
        if len(patches) < 2 or ridge <= 0 or not torch.isfinite(patches).all() or not torch.isfinite(targets).all():
            raise ValueError("finite training data, at least two rows, and positive ridge are required")
        pooled = torch.einsum("st,btd->bsd", self.pooling, patches)
        centered_x = pooled - pooled.mean(0)
        centered_y = targets - targets.mean(0)
        covariance = torch.cat([centered_x[:, s].T @ centered_y[:, s] for s in range(5)], dim=1)
        directions = torch.linalg.svd(covariance, full_matrices=False).U
        if self.latent_dim > directions.shape[1]:
            raise ValueError("latent width exceeds the supervised initialization rank")
        self.stem_weight.copy_(directions[:, :self.latent_dim])
        raw = pooled @ self.stem_weight
        standard = (raw - raw.mean(0)).flatten(0, 1).std(0).clamp_min(.1)
        self.stem_weight.div_(standard)
        self.stem_bias.copy_(-(pooled @ self.stem_weight).mean(0))
        hidden = self.stem(patches)
        for s in range(5):
            design = torch.cat([hidden[:, s], hidden[:, s].square()], dim=1)
            mean = design.mean(0)
            centered = design - mean
            weights = torch.linalg.solve(centered.T @ centered + ridge * torch.eye(2 * self.latent_dim),
                                         centered.T @ centered_y[:, s])
            self.linear_weight[s].copy_(weights[:self.latent_dim])
            self.square_weight[s].copy_(weights[self.latent_dim:])
            self.bias[s].copy_(targets[:, s].mean(0) - mean @ weights)

    def compile(self) -> CompiledSpatialQuadratic:
        model = copy.deepcopy(self).cpu().eval()
        with torch.enable_grad():
            x = torch.zeros(1, 16, self.patch_dim, requires_grad=True)
            h = model.stem(x).flatten()
            weight = torch.stack([
                torch.autograd.grad(h[i], x, retain_graph=i + 1 < h.numel())[0].flatten()
                for i in range(h.numel())
            ], dim=1).detach()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("circuit parameters must be finite")
        probes = torch.randn(2, 16, self.patch_dim, generator=torch.Generator().manual_seed(527))
        with torch.no_grad():
            reference = model.stem(probes).flatten(1)
            actual = probes.flatten(1) @ weight + h.detach()
        if not torch.allclose(reference, actual, atol=1e-4, rtol=1e-4):
            raise ValueError("spatial stem failed compilation parity")
        stem = CompiledAffineRelease(weight, h.detach(), (16, self.patch_dim), (5, self.latent_dim),
                                     float((reference - actual).abs().max()))
        return CompiledSpatialQuadratic(stem, model.linear_weight.detach(),
                                        model.square_weight.detach(), model.bias.detach())
