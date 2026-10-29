"""Deterministic anatomical slot layouts for chest radiograph releases.

The encrypted encoder still operates on a regular patch grid.  This module only
defines the public, fixed grouping used at the release boundary; it contains no
learned weights and therefore does not reveal a model checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class AnatomicalSlot:
    """A named release slot backed by a set of patch indices."""

    name: str
    token_indices: Tuple[int, ...]
    prompt_anchor: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("anatomical slot name must not be empty")
        if not self.prompt_anchor.strip():
            raise ValueError("anatomical slot prompt anchor must not be empty")
        if not self.token_indices:
            raise ValueError("anatomical slot must contain at least one token")
        if any(index < 0 for index in self.token_indices):
            raise ValueError("anatomical token indices must be non-negative")
        if len(set(self.token_indices)) != len(self.token_indices):
            raise ValueError("anatomical token indices must be unique")


CANONICAL_CXR_SLOT_NAMES = (
    "left_lung",
    "right_lung",
    "mediastinum_heart",
    "costophrenic_angles",
    "chest_wall_background",
)


def canonical_anatomical_slots(
    *, grid_size: int = 4, include_background: bool = True
) -> Tuple[AnatomicalSlot, ...]:
    """Return a deterministic five-slot layout for a square patch grid.

    The regions are intentionally coarse.  They are a public routing prior, not
    a learned segmentation mask, and should be reported as such in experiments.
    Every patch belongs to exactly one slot, which makes the release dimension
    and DP accountant unambiguous.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if not include_background:
        names = CANONICAL_CXR_SLOT_NAMES[:-1]
    else:
        names = CANONICAL_CXR_SLOT_NAMES

    slots: List[List[int]] = [[] for _ in names]
    for row in range(grid_size):
        for col in range(grid_size):
            index = row * grid_size + col
            # Left/right lung fields occupy the lateral columns.  The central
            # band contains heart/mediastinum, while the lowest row captures
            # costophrenic angles.  Remaining border patches are background.
            if row == grid_size - 1 and col in (0, grid_size - 1):
                slot = 3
            elif col < grid_size // 2 - (1 if grid_size >= 4 else 0):
                slot = 0
            elif col >= grid_size // 2 + (1 if grid_size >= 4 else 0):
                slot = 1
            elif row >= grid_size // 2:
                slot = 2
            else:
                slot = 4 if include_background else 2
            slots[slot].append(index)

    prompts = (
        "left lung",
        "right lung",
        "mediastinum and heart",
        "costophrenic angles",
        "chest wall and background",
    )
    result = tuple(
        AnatomicalSlot(name, tuple(indices), prompts[index])
        for index, (name, indices) in enumerate(zip(names, slots))
        if indices
    )
    validate_slot_partition(result, grid_size * grid_size)
    return result


def validate_slot_partition(
    slots: Sequence[AnatomicalSlot], num_tokens: int
) -> None:
    """Validate that slots partition ``range(num_tokens)`` exactly once."""
    if num_tokens < 1 or not slots:
        raise ValueError("a slot layout must contain at least one token")
    indices = [index for slot in slots for index in slot.token_indices]
    if sorted(indices) != list(range(num_tokens)):
        raise ValueError(
            "anatomical slots must partition every patch exactly once; "
            f"expected {num_tokens} indices, got {len(indices)}"
        )


def slot_token_groups(
    *, num_tokens: int = 16, grid_size: int = 4, include_background: bool = True
) -> List[List[int]]:
    """Return the canonical layout in the token-group format used by FDBR."""
    if num_tokens != grid_size * grid_size:
        raise ValueError("num_tokens must equal grid_size squared")
    return [list(slot.token_indices) for slot in canonical_anatomical_slots(
        grid_size=grid_size, include_background=include_background
    )]


__all__ = [
    "AnatomicalSlot",
    "CANONICAL_CXR_SLOT_NAMES",
    "canonical_anatomical_slots",
    "slot_token_groups",
    "validate_slot_partition",
]
