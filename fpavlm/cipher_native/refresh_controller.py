"""Runtime controller that turns LBR + PBSR plans into refresh actions.

FIRST PRINCIPLES (TenSEAL modulus-chain constraint)
  Any ciphertext that will be multiplied must be refreshed first. A CADN
  skip that still runs baked-affine + W_a + PolyAct + W_o + LTM exhausts
  the auto-rescale chain (observed: ``end of modulus switching chain``).

  Therefore a skipped LBR block is an IDENTITY SKIP: the block is not
  computed and those ciphertexts are never multiplied. PBSR background
  tokens are the same: they are copied through the residual and excluded
  from every cipher×plain / cipher×cipher op, including LTM region sums.

  LBR decides WHICH blocks are computed (and therefore refreshed).
  PBSR decides WHICH image tokens participate in a computed block.

  After adapter projection the token set is no longer anatomical patches, so
  PBSR applies only to the 16-token vision path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .lbr_scheduler import LBRSchedule
from .pbsr_scheduler import PBSRMask


@dataclass(frozen=True)
class RefreshAction:
    """What the encoder should do at one CADN boundary."""

    refresh: bool
    compute_block: bool
    token_indices: Optional[Tuple[int, ...]]
    use_baked_norm: bool
    rationale: str

    @property
    def n_tokens(self) -> Optional[int]:
        if self.token_indices is None:
            return None
        return len(self.token_indices)


class RefreshController:
    """Compose an LBR schedule with an optional PBSR mask.

    Args:
        schedule: LBR schedule. None means the historical full-CADN policy.
        mask: PBSR token mask. None means every token is refreshed.
        num_tokens: Image-token count used to expand a missing PBSR mask.
    """

    def __init__(
        self,
        schedule: Optional[LBRSchedule] = None,
        mask: Optional[PBSRMask] = None,
        num_tokens: int = 16,
    ):
        if num_tokens < 1:
            raise ValueError("num_tokens must be ≥ 1")
        self.schedule = schedule
        self.mask = mask
        self.num_tokens = num_tokens

    def action_before_block(self, block_idx: int) -> RefreshAction:
        """Refresh / compute decision immediately before ``block_idx`` runs."""
        refresh = True
        compute = True
        rationale = "full CADN (historical default)"
        if self.schedule is not None:
            refresh = self.schedule.uses_cadn_at(block_idx)
            compute = refresh
            rationale = (
                f"LBR CADN before block {block_idx}"
                if refresh
                else f"LBR identity-skip block {block_idx} (no multiply)"
            )
        # Block 0 always computes from a fresh CADN. Encrypt leftover levels
        # plus a full first block exhausted the TenSEAL chain in smoke tests.
        if block_idx == 0:
            refresh = True
            compute = True
            rationale = "forced CADN before block 0 (modulus-chain safety)"
        if not compute:
            return RefreshAction(
                refresh=False,
                compute_block=False,
                token_indices=None,
                use_baked_norm=False,
                rationale=rationale,
            )
        return RefreshAction(
            refresh=True,
            compute_block=True,
            token_indices=self._token_indices(),
            use_baked_norm=False,
            rationale=self._refresh_rationale(rationale),
        )

    def action_final_norm(self) -> RefreshAction:
        """Final-norm CADN is diagnostically critical; LBR never skips it."""
        return RefreshAction(
            refresh=True,
            compute_block=True,
            token_indices=self._token_indices(),
            use_baked_norm=False,
            rationale=self._refresh_rationale("final-norm CADN (always on)"),
        )

    def _token_indices(self) -> Optional[Tuple[int, ...]]:
        if self.mask is None:
            return None
        return self.mask.diagnostic_indices

    def _refresh_rationale(self, base: str) -> str:
        if self.mask is None:
            return base
        return (
            f"{base}; PBSR {len(self.mask.diagnostic_indices)}/"
            f"{self.mask.num_tokens} diagnostic tokens"
        )

    def expected_token_refreshes(self, num_blocks: int) -> int:
        """Count token-level CADN ops for the vision path (excludes adapter CANR)."""
        tokens = (
            len(self.mask.diagnostic_indices)
            if self.mask is not None
            else self.num_tokens
        )
        n = 1  # final norm
        for block_idx in range(num_blocks):
            if self.action_before_block(block_idx).refresh:
                n += 1
        return n * tokens


def select_ciphertexts(cipher_list: Sequence, indices: Optional[Sequence[int]]):
    """Return ``(indices, selected)`` for a refresh callback."""
    if indices is None:
        chosen = tuple(range(len(cipher_list)))
    else:
        chosen = tuple(int(i) for i in indices)
        for index in chosen:
            if index < 0 or index >= len(cipher_list):
                raise IndexError(f"PBSR token index {index} out of range {len(cipher_list)}")
    return chosen, [cipher_list[i] for i in chosen]


def merge_ciphertexts(cipher_list: Sequence, indices: Sequence[int], fresh: Sequence):
    """Replace selected positions with freshly refreshed ciphertexts."""
    if len(indices) != len(fresh):
        raise ValueError(
            f"refresh merge size mismatch: {len(indices)} indices vs {len(fresh)} ciphertexts"
        )
    merged = list(cipher_list)
    for index, ciphertext in zip(indices, fresh):
        merged[index] = ciphertext
    return merged


__all__ = [
    "RefreshAction",
    "RefreshController",
    "merge_ciphertexts",
    "select_ciphertexts",
]
