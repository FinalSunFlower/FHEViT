"""CiphertextTensor: PyTorch-compatible wrapper around TenSEAL CKKS ciphertext.

Provides a unified tensor-like API so cipher operations can be written as if on
plaintext, while actually executing on real CKKS ciphertexts via TenSEAL. This
enables true FHE inference with the same code structure as plaintext inference.

Key Design Decisions:
  1. Each CiphertextTensor wraps a list of TenSEAL CKKS vectors (batched).
  2. Arithmetic ops (+, *, scalar_mul) call TenSEAL directly.
  3. Operations track multiplicative depth for budget management.
  4. Falls back to plaintext torch.Tensor when engine is None (training mode).

Security: v50.2 removed the decrypt-reencrypt fallback in matmul_plain — the
server MUST NOT silently decrypt to compute matmul, as that leaks plaintext.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch


class CiphertextTensor:
    """Wraps TenSEAL CKKS ciphertext with PyTorch-like API.

    A CiphertextTensor can be in one of two modes:
      - CIPHER mode: wraps real TenSEAL CKKS vectors, ops are homomorphic
      - PLAIN mode: wraps a torch.Tensor, ops are plaintext (for training)

    This dual-mode design ensures train-inference consistency: the SAME code
    path runs in both modes, with only the underlying engine differing.
    """

    def __init__(
        self,
        data: Optional[torch.Tensor] = None,
        ciphertexts: Optional[List] = None,
        engine: Optional["object"] = None,
        depth: int = 0,
        raw_patch_layout: Optional[dict] = None,
    ):
        self._engine = engine
        self._depth = depth  # current multiplicative depth consumed
        self._plain: Optional[torch.Tensor] = None
        self._cipher: Optional[List] = None
        # Optional transport metadata used only before the server-side raw
        # patch projection.  It records how a high-dimensional patch was split
        # across CKKS ciphertexts; normal tensors leave it unset.
        self.raw_patch_layout = dict(raw_patch_layout) if raw_patch_layout else None

        if ciphertexts is not None:
            self._cipher = ciphertexts
            self._is_cipher = True
        elif data is not None:
            self._plain = data
            self._is_cipher = False
        else:
            raise ValueError("Must provide either data or ciphertexts")

    # ---- Properties ----
    @property
    def is_cipher(self) -> bool:
        return self._is_cipher

    def _require_plain(self) -> torch.Tensor:
        if self._plain is None:
            raise RuntimeError("CiphertextTensor invariant violated: plaintext is unavailable")
        return self._plain

    def _require_cipher(self) -> List:
        if self._cipher is None:
            raise RuntimeError("CiphertextTensor invariant violated: ciphertexts are unavailable")
        return self._cipher

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def shape(self):
        if self._is_cipher:
            if self._engine is not None and hasattr(self._engine, "_last_shape"):
                return self._engine._last_shape
            return None
        return self._require_plain().shape

    @property
    def dtype(self):
        if self._is_cipher:
            # TenSEAL CKKS encodes internally in float64; the exposed PyTorch
            # tensor view defaults to float32. Precision is governed by scale_bits.
            return torch.float32
        return self._require_plain().dtype

    @property
    def device(self):
        if self._is_cipher:
            return torch.device("cpu")  # TenSEAL runs on CPU
        return self._require_plain().device

    # ---- Arithmetic Operations (FHE primitives) ----
    def __add__(self, other: Union["CiphertextTensor", torch.Tensor, float, list]) -> "CiphertextTensor":
        """Ciphertext addition (0 depth)."""
        if isinstance(other, (int, float)):
            return self._scalar_add(float(other))
        if isinstance(other, list):
            other = torch.tensor(other, dtype=torch.float32)
        if isinstance(other, torch.Tensor):
            other = CiphertextTensor(data=other, engine=self._engine, depth=self._depth)
        if not self._is_cipher and not other._is_cipher:
            return CiphertextTensor(
                data=self._require_plain() + other._require_plain(),
                engine=self._engine,
                depth=max(self._depth, other._depth),
            )
        return self._cipher_add(other)

    def __sub__(self, other: Union["CiphertextTensor", torch.Tensor, float, list]) -> "CiphertextTensor":
        """Ciphertext subtraction (0 depth).

        a - b = a + (-b). In CKKS, subtraction is 0 depth (same as addition).
        Supports: cipher - cipher, cipher - scalar, cipher - plain_list, plain - plain.
        """
        if isinstance(other, (int, float)):
            return self._scalar_add(-float(other))  # a - s = a + (-s), 0 depth
        if isinstance(other, list):
            other = torch.tensor(other, dtype=torch.float32)
        if isinstance(other, torch.Tensor):
            # cipher - plain = cipher + (-plain); plain - plain = plain - plain
            if not self._is_cipher:
                return CiphertextTensor(
                    data=self._require_plain() - other,
                    engine=self._engine,
                    depth=self._depth,
                )
            plain_neg = (-other).detach().cpu().reshape(-1).tolist()
            new_ct = [ct + plain_neg for ct in self._require_cipher()]
            return CiphertextTensor(ciphertexts=new_ct, engine=self._engine, depth=self._depth)
        # other is a CiphertextTensor — handle plain-plain and cipher-cipher
        if not self._is_cipher and not other._is_cipher:
            return CiphertextTensor(
                data=self._require_plain() - other._require_plain(),
                engine=self._engine,
                depth=max(self._depth, other._depth),
            )
        # At least one is cipher — dispatch to cipher path (handles cipher-cipher
        # and the rare plain-cipher case via _cipher_sub's None-guard).
        return self._cipher_sub(other)

    def __neg__(self) -> "CiphertextTensor":
        """Negation (0 depth in CKKS — just negates polynomial coefficients)."""
        if not self._is_cipher:
            return CiphertextTensor(data=-self._require_plain(), engine=self._engine, depth=self._depth)
        new_ct = [-ct for ct in self._require_cipher()]
        return CiphertextTensor(ciphertexts=new_ct, engine=self._engine, depth=self._depth)

    def _cipher_sub(self, other: "CiphertextTensor") -> "CiphertextTensor":
        """cipher - cipher / cipher - plain / plain - cipher: 0 depth.

        CKKS subtraction = addition of negated ciphertext (0 depth).
        Handles three cases (the dispatch in __sub__ routes any case where at
        least one operand is cipher here):
          - cipher - cipher: ct_a - ct_b (TenSEAL native, 0 depth)
          - cipher - plain:  ct + (-plain_list) (0 depth)
          - plain - cipher:  -(cipher - plain) = -ct + plain_list (0 depth)
        """
        # plain - cipher: invert to -(cipher - plain) to keep self as the cipher.
        # Without this guard, `for ct in self._cipher` would crash (self._cipher is None).
        if self._cipher is None:
            other_cipher = other._require_cipher()
            # plain - cipher = -(cipher - plain) = negate(cipher) + plain
            plain_list = self._require_plain().detach().cpu().reshape(-1).tolist()
            new_ct = [(-ct) + plain_list for ct in other_cipher]
            return CiphertextTensor(
                ciphertexts=new_ct, engine=other._engine, depth=other._depth,
            )
        # cipher - plain: ct + (-plain_list)
        if other._cipher is None:
            plain_neg = (-other._require_plain()).detach().cpu().reshape(-1).tolist()
            new_ct = [ct + plain_neg for ct in self._require_cipher()]
            return CiphertextTensor(ciphertexts=new_ct, engine=self._engine, depth=self._depth)
        # cipher - cipher: TenSEAL native subtraction
        self_cipher = self._require_cipher()
        other_cipher = other._require_cipher()
        if len(self_cipher) != len(other_cipher):
            raise ValueError(f"Ciphertext count mismatch: {len(self_cipher)} vs {len(other_cipher)}")
        new_ct = [a - b for a, b in zip(self_cipher, other_cipher)]
        return CiphertextTensor(
            ciphertexts=new_ct,
            engine=self._engine,
            depth=max(self._depth, other._depth),
        )

    def __mul__(self, other: Union["CiphertextTensor", torch.Tensor, float, list]) -> "CiphertextTensor":
        """Multiplication. ALL multiplications consume +1 depth in TenSEAL CKKS.

        EMPIRICAL FINDING (TenSEAL CKKS micro-benchmark):
          In TenSEAL CKKS with auto_rescale=True, ALL multiplications consume
          1 depth level — cipher×cipher, cipher×plain, AND cipher×scalar.
          auto_rescale rescales after every multiplication to keep the scale
          bounded, and each rescale removes one prime from the modulus chain
          (i.e., consumes one "level").

          This contradicts the theoretical CKKS property where cipher×plain
          could avoid rescaling if the plain is encoded at scale 1. TenSEAL's
          auto_rescale is conservative and always rescales.

          Implication: depth tracking must increment +1 for EVERY multiplication.
          The architecture's original assumption of "cipher×plain = 0 depth"
          was WRONG and caused "scale out of bounds" errors when the actual
          depth silently exceeded the budget.
        """
        if isinstance(other, (int, float)):
            return self._scalar_mul(float(other))
        if isinstance(other, list):
            other = torch.tensor(other, dtype=torch.float32)
        if isinstance(other, torch.Tensor):
            return self._plain_mul(other)
        if isinstance(other, CiphertextTensor):
            if not self._is_cipher and not other._is_cipher:
                return CiphertextTensor(
                    data=self._require_plain() * other._require_plain(),
                    engine=self._engine,
                    depth=max(self._depth, other._depth) + 1,
                )
            return self._cipher_mul(other)
        raise TypeError(f"Unsupported type for mul: {type(other)}")

    def _scalar_add(self, s: float) -> "CiphertextTensor":
        if not self._is_cipher:
            return CiphertextTensor(data=self._require_plain() + s, engine=self._engine, depth=self._depth)
        new_ct = [ct + s for ct in self._require_cipher()]
        return CiphertextTensor(ciphertexts=new_ct, engine=self._engine, depth=self._depth)

    def _scalar_mul(self, s: float) -> "CiphertextTensor":
        """cipher × scalar: +1 depth (TenSEAL auto_rescale consumes a level)."""
        if not self._is_cipher:
            return CiphertextTensor(data=self._require_plain() * s, engine=self._engine, depth=self._depth)
        new_ct = [ct * s for ct in self._require_cipher()]
        return CiphertextTensor(ciphertexts=new_ct, engine=self._engine, depth=self._depth + 1)

    def _plain_mul(self, plain: torch.Tensor) -> "CiphertextTensor":
        """cipher × plain: +1 depth (TenSEAL auto_rescale consumes a level)."""
        if not self._is_cipher:
            return CiphertextTensor(
                data=self._require_plain() * plain,
                engine=self._engine,
                depth=self._depth,
            )
        plain_list = plain.detach().cpu().reshape(-1).tolist()
        new_ct = []
        for ct in self._require_cipher():
            slot_count = ct.size() if hasattr(ct, "size") else len(plain_list)
            # v175 FIX: previously `plain_list[:slot_count]` silently truncated
            # when len(plain_list) > slot_count, producing wrong results without
            # any error. Now raise so the caller fixes the input shape.
            if len(plain_list) > slot_count:
                raise ValueError(
                    f"Plain vector length {len(plain_list)} exceeds CKKS slot "
                    f"count {slot_count}. The plain operand must fit in one "
                    f"ciphertext slot (broadcast semantics). If you need to "
                    f"multiply by a longer vector, split it across multiple "
                    f"CiphertextTensors explicitly."
                )
            chunk = plain_list[:slot_count]
            new_ct.append(ct * chunk)
        return CiphertextTensor(ciphertexts=new_ct, engine=self._engine, depth=self._depth + 1)

    def _cipher_add(self, other: "CiphertextTensor") -> "CiphertextTensor":
        if other._cipher is None:
            plain_list = other._require_plain().detach().cpu().reshape(-1).tolist()
            new_ct = [ct + plain_list for ct in self._require_cipher()]
            return CiphertextTensor(
                ciphertexts=new_ct, engine=self._engine, depth=self._depth)
        self_cipher = self._require_cipher()
        other_cipher = other._require_cipher()
        if len(self_cipher) != len(other_cipher):
            raise ValueError(f"Ciphertext count mismatch: {len(self_cipher)} vs {len(other_cipher)}")
        new_ct = [a + b for a, b in zip(self_cipher, other_cipher)]
        return CiphertextTensor(
            ciphertexts=new_ct,
            engine=self._engine,
            depth=max(self._depth, other._depth),
        )

    def _cipher_mul(self, other: "CiphertextTensor") -> "CiphertextTensor":
        """cipher × cipher: +1 depth. cipher × plain (fallback): +1 depth."""
        if other._cipher is None:
            plain_list = other._require_plain().detach().cpu().reshape(-1).tolist()
            new_ct = []
            for ct in self._require_cipher():
                slot_count = ct.size() if hasattr(ct, "size") else len(plain_list)
                # v175 FIX: raise on truncation (see _plain_mul for rationale).
                if len(plain_list) > slot_count:
                    raise ValueError(
                        f"Plain vector length {len(plain_list)} exceeds CKKS slot "
                        f"count {slot_count}. Split across multiple CiphertextTensors."
                    )
                chunk = plain_list[:slot_count]
                new_ct.append(ct * chunk)
            return CiphertextTensor(
                ciphertexts=new_ct, engine=self._engine, depth=self._depth + 1)
        self_cipher = self._require_cipher()
        other_cipher = other._require_cipher()
        if len(self_cipher) != len(other_cipher):
            raise ValueError(f"Ciphertext count mismatch: {len(self_cipher)} vs {len(other_cipher)}")
        new_ct = [a * b for a, b in zip(self_cipher, other_cipher)]
        return CiphertextTensor(
            ciphertexts=new_ct,
            engine=self._engine,
            depth=max(self._depth, other._depth) + 1,
        )

    # ---- Linear algebra helpers ----
    def matmul_plain(self, weight: torch.Tensor) -> "CiphertextTensor":
        """Matrix multiplication with plaintext weight: out = x @ W.

        For cipher mode: dispatches to engine.matmul_plain (real CKKS matmul
        via diagonal packing or TenSEAL built-in). For plain mode: torch.matmul.

        Depth: +1 (TenSEAL ct.matmul consumes a level via auto_rescale).

        v50.2 SECURITY FIX: There is NO decrypt-reencrypt fallback. The server
        MUST NOT silently decrypt to compute matmul, as that leaks plaintext.
        Callers must provide an engine implementing `matmul_plain` (e.g.,
        ServerCANREngine); otherwise this raises RuntimeError.
        """
        if not self._is_cipher:
            return CiphertextTensor(
                data=self._require_plain() @ weight,
                engine=self._engine,
                depth=self._depth,
            )
        if self._engine is not None and hasattr(self._engine, "matmul_plain"):
            new_ct = self._engine.matmul_plain(self._require_cipher(), weight.detach().cpu())
            return CiphertextTensor(ciphertexts=new_ct, engine=self._engine, depth=self._depth + 1)
        raise RuntimeError(
            "CiphertextTensor.matmul_plain: no FHE engine with matmul_plain support. "
            "Decrypt-reencrypt fallback removed in v50.2 for security. "
            "Provide an engine implementing matmul_plain (e.g., ServerCANREngine)."
        )

    def pool_tokens(self, groups: list[list[int]]) -> "CiphertextTensor":
        """Mean-pool disjoint token groups using only CKKS additions/scalars."""
        if not groups or any(not group for group in groups):
            raise ValueError("token groups must be non-empty")
        flattened = [index for group in groups for index in group]
        token_count = len(self._require_cipher()) if self._is_cipher else self._require_plain().shape[-2]
        if sorted(flattened) != list(range(token_count)):
            raise ValueError("token groups must partition every input token exactly once")
        if not self._is_cipher:
            pooled = torch.stack(
                [self._require_plain()[..., group, :].mean(dim=-2) for group in groups],
                dim=-2,
            )
            return CiphertextTensor(data=pooled, engine=self._engine, depth=self._depth)
        pooled_ciphertexts = []
        pooled_depth = self._depth
        for group in groups:
            cipher = self._require_cipher()
            value = cipher[group[0]]
            for index in group[1:]:
                value = value + cipher[index]
            if len(group) > 1:
                value = value * (1.0 / len(group))
                pooled_depth = self._depth + 1
            pooled_ciphertexts.append(value)
        return CiphertextTensor(
            ciphertexts=pooled_ciphertexts,
            engine=self._engine,
            depth=pooled_depth,
        )

    # ---- Utility ----
    def __repr__(self):
        mode = "CIPHER" if self._is_cipher else "PLAIN"
        if self._is_cipher:
            return f"CiphertextTensor(mode={mode}, n_ct={len(self._require_cipher())}, depth={self._depth})"
        return f"CiphertextTensor(mode={mode}, shape={self._require_plain().shape}, depth={self._depth})"
