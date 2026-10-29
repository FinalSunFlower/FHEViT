"""Exact compilation of the supported affine vision circuit into its release.

This is constant folding, not a new privacy guarantee or a nonlinear ViT.
The private compiled matrix stays on the server. Only raw-patch ciphertexts
and encrypted diagnostic coefficients cross the boundary. Nonlinear students
must use the original polynomial executor.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from .cipher_native.cipher_norm import CipherRMSNorm
from .cipher_native.diagnostic_bottleneck import DiagnosticBottleneckRelease
from .cipher_native.med_fhe_vit_block import AnatomyStructuredTokenMixer, MedFHEBlock
from .cipher_native.noise_resilient_encoder import NoiseResilientMAViTEncoder
from .cipher_native.poly_norm import PolyNorm
from .cipher_native.spectral_bounded import SpectralBoundedLinear


@dataclass(frozen=True)
class CompiledAffineRelease:
    """Server-private snapshot: flattened raw patches @ weight + bias."""

    weight: torch.Tensor
    bias: torch.Tensor
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    reference_max_abs_error: float

    @property
    def multiplicative_depth(self) -> int:
        return 1

    def plain(self, patches: torch.Tensor) -> torch.Tensor:
        if tuple(patches.shape[-2:]) != self.input_shape:
            raise ValueError("raw patch shape does not match compiled circuit")
        if not torch.isfinite(patches).all():
            raise ValueError("raw patches must be finite")
        x = patches.reshape(-1, self.weight.shape[0]).to(self.weight)
        return (x @ self.weight + self.bias).reshape(-1, *self.output_shape)

    def evaluate(self, ciphertexts: list, *, chunk_size: int) -> list:
        """Dot/reduce on public-context vectors; no decrypt or callback.

        Each output coordinate is a scalar ciphertext. Dot reduction uses
        rotations/additions after one plaintext multiplication. This avoids
        materializing every intermediate token and the private Perceiver.
        """
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        inputs = self.weight.shape[0]
        if len(ciphertexts) != (inputs + chunk_size - 1) // chunk_size:
            raise ValueError("ciphertext count does not match raw-patch layout")
        for index, ct in enumerate(ciphertexts):
            if ct.context().has_secret_key():
                raise ValueError("server ciphertext context must not contain a secret key")
            if ct.size() != min(chunk_size, inputs - index * chunk_size):
                raise ValueError("ciphertext vector length does not match raw-patch layout")
        outputs = []
        for column in range(self.weight.shape[1]):
            total = None
            for index, ct in enumerate(ciphertexts):
                start = index * chunk_size
                w = self.weight[start:start + chunk_size, column].tolist()
                # An all-zero plaintext product is a transparent ciphertext
                # rejected by SEAL. Skip it without introducing a fake epsilon.
                if not any(w):
                    continue
                partial = ct.dot(w)
                total = partial if total is None else total + partial
            if total is None:
                import tenseal as ts

                total = ts.ckks_vector(ciphertexts[0].context(), [0.0])
            outputs.append(total + float(self.bias[column]))
        return outputs


def compile_affine_release(
    encoder: NoiseResilientMAViTEncoder,
    bottleneck: DiagnosticBottleneckRelease,
) -> CompiledAffineRelease:
    """Compile only known degree-1 operators, with independent parity probes.

    For an affine function f, J=f' is constant and f(x)=Jx+f(0). Computing
    one reverse pass per released scalar avoids a square pixel-space matrix.
    Compilation is offline; no patient data or secret key is used.
    """
    if type(encoder) is not NoiseResilientMAViTEncoder or encoder.use_cadn:
        raise ValueError("compilation requires the built-in non-CADN encoder")
    if type(bottleneck) is not DiagnosticBottleneckRelease:
        raise ValueError("compilation requires the built-in diagnostic release")
    if type(encoder.patch_proj) is not SpectralBoundedLinear or type(encoder.final_norm) is not CipherRMSNorm:
        raise ValueError("unsupported patch projection or final normalization")
    for block in encoder.blocks:
        if (
            type(block) is not MedFHEBlock
            or block.use_cadn
            or block.activation_type != "polyact"
            or block.poly_act_degree != 1
            or type(block.norm1) is not PolyNorm
            or block.norm1.poly_degree != 1
            or (block.token_mixer is not None and type(block.token_mixer) is not AnatomyStructuredTokenMixer)
        ):
            raise ValueError("nonlinear or unsupported block cannot be compiled as affine")
    if any(module._forward_hooks or module._forward_pre_hooks for module in encoder.modules()):
        raise ValueError("forward hooks are not supported by the affine compiler")

    # Baked affine caches can be non-leaf tensors; snapshot them explicitly
    # instead of mutating or clearing the caller's inference caches.
    memo = {}
    visited = set()

    def snapshot_caches(value):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, torch.Tensor):
            if not value.is_leaf:
                memo[id(value)] = value.detach().clone()
        elif isinstance(value, dict):
            for child in value.values():
                snapshot_caches(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                snapshot_caches(child)
        elif isinstance(value, torch.nn.Module):
            snapshot_caches(vars(value))

    snapshot_caches(encoder)
    model = copy.deepcopy(encoder, memo).cpu().eval()
    # Compilation stops before the private nonlinear bridge. Removing it also
    # restores final normalization when the ordinary adapter had fused it.
    model.concept_adapter = None
    model.fuse_final_affine = False
    model.bake_statistics()
    for block in model.blocks:
        quadratic = block.norm1.c2 if block.norm1.num_regions == 1 else block.norm1.c2_regions
        if torch.count_nonzero(quadratic):
            raise ValueError("degree-1 declaration has nonzero quadratic normalization coefficients")
    input_shape = (model.num_patches, model.patch_proj.weight.shape[0])

    def reference(x):
        return bottleneck.encode_plain(model.forward_plain_fhe_match(x))

    with torch.enable_grad():
        origin = torch.zeros(1, *input_shape, requires_grad=True)
        values = reference(origin)
        output_shape = tuple(values.shape[-2:])
        flat = values.flatten()
        columns = [
            torch.autograd.grad(flat[i], origin, retain_graph=i + 1 < flat.numel())[0].flatten()
            for i in range(flat.numel())
        ]
    weight = torch.stack(columns, dim=1).detach().contiguous()
    bias = flat.detach().clone()
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("compiled coefficients must be finite")
    generator = torch.Generator().manual_seed(781)
    probes = torch.randn(3, *input_shape, generator=generator) * model.blocks[0].input_range
    with torch.no_grad():
        expected = reference(probes)
        actual = (probes.flatten(1) @ weight + bias).reshape_as(expected)
    if not torch.allclose(actual, expected, rtol=2e-4, atol=2e-4):
        raise ValueError("compiled circuit failed independent affine parity probes")
    return CompiledAffineRelease(
        weight, bias, input_shape, output_shape, float((actual - expected).abs().max()),
    )
