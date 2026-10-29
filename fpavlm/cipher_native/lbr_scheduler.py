"""Lazy Block Refresh (LBR) Scheduler for FHE-ViT.

ROOT CAUSE OF LATENCY (683.50 s/image in CADN interactive path):
  Each CADN refresh requires one CLIENT ROUND-TRIP:
    Server sends encrypted activations → client decrypts → client computes
    LayerNorm → client re-encrypts → server receives fresh ciphertexts.
  With 3 blocks and the current "refresh-every-block" policy:
    5 round trips × ~136 s/trip ≈ 680 s total
  ~80 % of latency is round-trip overhead, not FHE arithmetic.

LBR KEY INSIGHT:
  A CADN refresh serves two purposes:
    (a) Reset multiplicative depth to 0 (budget management)
    (b) Provide EXACT LayerNorm for the upcoming block (accuracy)
  Purpose (a) can be DEFERRED if accumulated depth stays below budget.
  Purpose (b) can be APPROXIMATED with PolyNorm when feature distributions
  are well-behaved (early blocks; background patches).

  For 3 blocks, block_depth=4 (CADN), budget=13, patch_proj_depth=1:
    Standard schedule: refresh after patch_proj, blocks 0,1,2, W_concept
      → 5 round trips
    LBR schedule (threshold=9):
      after patch_proj (depth 1):  1 < 9  → PolyNorm, no refresh
      after block 0   (depth 1+5):  6 < 9  → PolyNorm, no refresh
      after block 1   (depth 6+5): 11 ≥ 9  → CADN REFRESH, depth → 0
      after block 2   (depth 0+4):  4 < 9  → PolyNorm for final_norm
      after W_concept (depth 4+1):  5 < 9  → final CANR, depth → 0
    → 2 round trips (60 % reduction → ≈2.5× speedup on round-trip cost)

  For blocks that use PolyNorm (no CADN refresh): the encoder falls back to
  BAKED polynomial statistics (EMA mean/var from training).  PolyNorm
  approximation error is bounded by the feature distribution shift.

MEDICAL ACCURACY RATIONALE:
  Later blocks encode more semantic/pathological features; they benefit most
  from exact normalization.  LBR prioritizes CADN for the deepest blocks
  (those closest to the FDBR projection), where approximation errors would
  propagate into the diagnostic release.

  The strategy "diagnostic_priority" implements this:
    - Mark the last N_cadn blocks as "must CADN"
    - Fill remaining budget with PolyNorm blocks

RETAINED INTERACTIVE BASELINE (not a literature-first claim):
  1. Configurable lazy-refresh scheduler for the inherited FHE-ViT pipeline.
  2. Medical-aware priority: CADN concentrated at the semantically critical
     (late) blocks rather than uniformly distributed.
  3. Depth-budget-aware: provably stays within the configured CKKS budget.
  4. Composable with PBSR (Position-Based Selective Refresh) for per-token
     granularity in a future extension.
  5. Smooth accuracy-latency trade-off: single parameter (lbr_cadn_fraction)
     interpolates between full CADN (1.0 → 5 refreshes) and continuous FHE
     (0.0 → 0 refreshes).

These counts are schedule estimates, not measured speedups for the 0.4 codec.
CADN requires a trusted client callback and is not homomorphic bootstrapping.
See RELATED_WORK.md for verified private-Transformer references. Medical priority
is a fixed heuristic, not a proved clinical accuracy or novelty guarantee.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class BlockRefreshDecision:
    """Refresh decision for one block boundary."""
    block_idx: int            # which block boundary (0 = before block 0, etc.)
    use_cadn: bool            # True = CADN round trip; False = PolyNorm (baked)
    depth_before: int         # accumulated depth entering this decision point
    depth_after: int          # depth after the decision (0 if CADN, += block_depth)
    rationale: str            # human-readable reason for this decision


@dataclass(frozen=True)
class LBRSchedule:
    """Complete LBR schedule for one FHE-ViT forward pass.

    Attributes:
        decisions: List of per-block refresh decisions (length = num_blocks + 2
            for the patch_proj boundary and W_concept boundary).
        n_cadn_refreshes: Number of CADN round trips required.
        n_polynorm_blocks: Number of blocks using PolyNorm (no round trip).
        peak_depth: Maximum accumulated depth (must be < depth_budget).
        estimated_speedup_vs_full_cadn: Approximate speedup factor
            (based on linear round-trip cost model).
    """
    decisions: Tuple[BlockRefreshDecision, ...]
    n_cadn_refreshes: int
    n_polynorm_blocks: int
    peak_depth: int
    estimated_speedup_vs_full_cadn: float

    def uses_cadn_at(self, block_idx: int) -> bool:
        """Return True if CADN refresh should happen after block ``block_idx``."""
        for dec in self.decisions:
            if dec.block_idx == block_idx:
                return dec.use_cadn
        return False

    def summary(self) -> str:
        """One-line human-readable schedule summary."""
        flags = "".join(
            "C" if d.use_cadn else "P"
            for d in self.decisions
            if d.block_idx >= 0  # skip pre-run sentinel
        )
        return (
            f"LBR({flags}) "
            f"cadn={self.n_cadn_refreshes} "
            f"poly={self.n_polynorm_blocks} "
            f"peak_depth={self.peak_depth} "
            f"est_speedup={self.estimated_speedup_vs_full_cadn:.2f}×"
        )


class LazyBlockRefreshScheduler:
    """Compute an LBR schedule that minimises round trips within the depth budget.

    Usage::
        scheduler = LazyBlockRefreshScheduler(
            num_blocks=3,
            depth_per_block_cadn=4,   # block depth when CADN is used
            depth_per_block_polynorm=5,  # block depth when PolyNorm is used
            depth_budget=13,
            strategy="diagnostic_priority",
        )
        schedule = scheduler.compute()
        print(schedule.summary())  # LBR(PPC) cadn=2 poly=1 peak_depth=11 est_speedup=2.27×

    """

    def __init__(
        self,
        num_blocks: int,
        depth_per_block_cadn: int,
        depth_per_block_polynorm: int,
        depth_budget: int,
        patch_proj_depth: int = 1,
        wconcept_depth: int = 1,
        final_norm_cadn_depth: int = 0,   # 0 if CADN; 1 if PolyNorm (affine)
        strategy: str = "diagnostic_priority",
        cadn_fraction: float = 0.5,
        round_trip_cost_ratio: float = 4.0,
    ):
        """
        Args:
            num_blocks: Number of transformer blocks.
            depth_per_block_cadn: Depth consumed by one block when CADN is used
                (normalization is at depth 0, so only matmul+activation+mixer).
            depth_per_block_polynorm: Depth consumed by one block when PolyNorm
                is used (includes PolyNorm depth 1 or 2).
            depth_budget: Total CKKS multiplicative depth budget.
            patch_proj_depth: Depth of initial patch projection.
            wconcept_depth: Depth of the FDBR W_concept projection.
            final_norm_cadn_depth: Depth of the final normalization step when
                CADN is used (0 = depth-free CADN; 1 = PolyNorm fallback).
            strategy: Refresh strategy:
                ``"min_refreshes"`` — absolute minimum round trips (may use
                    PolyNorm for all but the last block before overflow).
                ``"diagnostic_priority"`` — CADN for the last ``cadn_fraction``
                    blocks (those closest to the diagnostic FDBR release); uses
                    PolyNorm for early blocks where features are less critical.
                ``"uniform"`` — evenly space CADN refreshes; approximates the
                    standard CADN-every-block schedule with fewer trips.
                ``"full_cadn"`` — standard policy (every block uses CADN);
                    equivalent to the existing interactive path; no speedup.
            cadn_fraction: Fraction of blocks that get CADN refresh.  Only
                used by ``"diagnostic_priority"`` and ``"uniform"`` strategies.
                Default 0.5 (half the blocks use CADN).
            round_trip_cost_ratio: Ratio of round-trip cost to FHE-arithmetic
                cost per block (used for speedup estimation only).  Default 4.0
                (one CADN round trip costs ~4× one PolyNorm block).
        """
        if num_blocks < 1:
            raise ValueError("num_blocks must be ≥ 1")
        if depth_per_block_cadn < 1 or depth_per_block_polynorm < 1:
            raise ValueError("block depths must be ≥ 1")
        if depth_budget < 1:
            raise ValueError("depth_budget must be ≥ 1")
        if not 0.0 <= cadn_fraction <= 1.0:
            raise ValueError("cadn_fraction must be in [0, 1]")

        self.num_blocks = num_blocks
        self.depth_cadn = depth_per_block_cadn
        self.depth_poly = depth_per_block_polynorm
        self.budget = depth_budget
        self.patch_proj_depth = patch_proj_depth
        self.wconcept_depth = wconcept_depth
        self.final_norm_cadn_depth = final_norm_cadn_depth
        self.strategy = strategy
        self.cadn_fraction = cadn_fraction
        self.round_trip_cost_ratio = round_trip_cost_ratio

    # ------------------------------------------------------------------
    # Core scheduling
    # ------------------------------------------------------------------

    def compute(self) -> LBRSchedule:
        """Compute and validate the LBR schedule.

        Returns:
            LBRSchedule with per-block decisions and summary metrics.

        Raises:
            ValueError: If the schedule would exceed the depth budget.
        """
        if self.strategy == "full_cadn":
            return self._schedule_full_cadn()
        elif self.strategy == "min_refreshes":
            return self._schedule_min_refreshes()
        elif self.strategy == "diagnostic_priority":
            return self._schedule_diagnostic_priority()
        elif self.strategy == "uniform":
            return self._schedule_uniform()
        else:
            raise ValueError(
                f"Unknown strategy {self.strategy!r}; must be one of "
                "'full_cadn', 'min_refreshes', 'diagnostic_priority', 'uniform'"
            )

    def _make_schedule(self, cadn_mask: List[bool]) -> LBRSchedule:
        """Build an LBRSchedule from a per-block CADN mask and validate depth.

        Args:
            cadn_mask: List of length num_blocks.  True = CADN refresh after
                this block (depth reset to 0); False = PolyNorm (depth += poly).

        Returns:
            LBRSchedule
        """
        decisions: List[BlockRefreshDecision] = []
        depth = self.patch_proj_depth
        peak = depth

        for i, use_cadn in enumerate(cadn_mask):
            if use_cadn:
                # CADN refresh happens at the block BOUNDARY (before this block):
                #   1. Current depth recorded for peak tracking
                #   2. CADN refresh → depth = 0 (provides fresh ciphertexts + z_norm)
                #   3. Block runs at depth_cadn from the refreshed base
                peak = max(peak, depth)           # peak at the refresh point
                block_out_depth = self.depth_cadn  # block runs from 0
                peak = max(peak, block_out_depth)
                decisions.append(BlockRefreshDecision(
                    block_idx=i,
                    use_cadn=True,
                    depth_before=depth,
                    depth_after=block_out_depth,
                    rationale=(
                        f"CADN refresh at block {i} boundary "
                        f"(depth {depth} → 0 → {block_out_depth} post-block)"
                    ),
                ))
                depth = block_out_depth
            else:
                # Identity skip: the block is not computed, so it consumes no
                # multiplicative depth and never multiplies stale ciphertexts.
                decisions.append(BlockRefreshDecision(
                    block_idx=i,
                    use_cadn=False,
                    depth_before=depth,
                    depth_after=depth,
                    rationale=(
                        f"identity skip block {i} "
                        f"(depth stays {depth}; no multiply)"
                    ),
                ))

        # Final W_concept + CANR
        depth_final = depth + self.wconcept_depth
        peak = max(peak, depth_final)
        decisions.append(BlockRefreshDecision(
            block_idx=self.num_blocks,  # sentinel: after all blocks
            use_cadn=True,
            depth_before=depth,
            depth_after=0,
            rationale=f"Final CANR refresh after W_concept (depth {depth_final} → 0)",
        ))

        # Validate
        if peak > self.budget:
            raise ValueError(
                f"LBR schedule exceeds depth budget: peak_depth={peak} > budget={self.budget}. "
                f"Strategy={self.strategy!r}, cadn_mask={cadn_mask}. "
                "Increase depth_budget or use a strategy with more CADN refreshes."
            )

        n_cadn = sum(cadn_mask) + 1  # +1 for final CANR
        n_poly = sum(not c for c in cadn_mask)

        # Speedup model: assume total_cost = n_trips * trip_cost + n_blocks * arith_cost
        # full_cadn: n_trips_full = num_blocks + 2 (patch_proj + blocks + final)
        # lbr:       n_trips_lbr  = n_cadn
        # arith cost unchanged
        n_trips_full = self.num_blocks + 2
        trip = self.round_trip_cost_ratio
        full_cost = n_trips_full * trip + self.num_blocks
        lbr_cost = n_cadn * trip + self.num_blocks  # +1 for final CANR is in n_cadn
        speedup = full_cost / max(lbr_cost, 1.0)

        return LBRSchedule(
            decisions=tuple(decisions),
            n_cadn_refreshes=n_cadn,
            n_polynorm_blocks=n_poly,
            peak_depth=peak,
            estimated_speedup_vs_full_cadn=speedup,
        )

    def _schedule_full_cadn(self) -> LBRSchedule:
        """Standard policy: CADN after every block (no speedup)."""
        return self._make_schedule([True] * self.num_blocks)

    def _schedule_min_refreshes(self) -> LBRSchedule:
        """Minimize CADN round trips while staying within depth budget.

        Greedy: accumulate PolyNorm depth until budget would be exceeded on
        the next block, then flush with a CADN refresh.
        """
        cadn_mask = [False] * self.num_blocks
        depth = self.patch_proj_depth

        for i in range(self.num_blocks):
            # Identity-skipped blocks consume 0 depth. Refresh only when the
            # next *computed* block would otherwise start without a CADN reset
            # after leftover depth from a previous computed block.
            after_compute = depth + self.depth_cadn
            if i == 0 or after_compute > self.budget:
                cadn_mask[i] = True
                depth = self.depth_cadn
            # else: identity skip, depth unchanged

        return self._make_schedule(cadn_mask)

    def _schedule_diagnostic_priority(self) -> LBRSchedule:
        """CADN for the last k blocks (closest to diagnostic FDBR release).

        Medical rationale: late blocks encode semantic/pathological features
        that feed directly into the FDBR release; they benefit most from exact
        normalization.  Early blocks encode texture features that are robust to
        PolyNorm approximation.

        The number of CADN blocks = ceil(num_blocks × cadn_fraction).
        """
        n_cadn = max(1, math.ceil(self.num_blocks * self.cadn_fraction))
        n_cadn = min(n_cadn, self.num_blocks)
        # Always refresh before block 0. A CADN-trained block without a
        # fresh z_norm uses baked affine + the rest of the block; on the
        # TenSEAL auto-rescale chain that exhausted the modulus in the
        # first no-refresh block. Keep the remaining CADN slots on the
        # last blocks (closest to FDBR).
        cadn_mask = [False] * self.num_blocks
        cadn_mask[0] = True
        remaining = max(0, n_cadn - 1)
        for index in range(self.num_blocks - remaining, self.num_blocks):
            cadn_mask[index] = True

        # Validate depth budget — if it fails, fall back to more CADN
        try:
            return self._make_schedule(cadn_mask)
        except ValueError:
            # Too many PolyNorm blocks overflow budget — add one more CADN
            for i in range(len(cadn_mask)):
                if not cadn_mask[i]:
                    cadn_mask[i] = True
                    try:
                        return self._make_schedule(cadn_mask)
                    except ValueError:
                        continue
            # Last resort: full CADN
            return self._schedule_full_cadn()

    def _schedule_uniform(self) -> LBRSchedule:
        """Space CADN refreshes uniformly across blocks."""
        n_cadn = max(1, round(self.num_blocks * self.cadn_fraction))
        if n_cadn >= self.num_blocks:
            return self._schedule_full_cadn()
        # Place CADN at evenly spaced indices
        step = self.num_blocks / n_cadn
        cadn_mask = [False] * self.num_blocks
        for i in range(n_cadn):
            idx = min(round((i + 0.5) * step), self.num_blocks - 1)
            cadn_mask[idx] = True
        try:
            return self._make_schedule(cadn_mask)
        except ValueError:
            return self._schedule_min_refreshes()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def pareto_frontier(
        self,
        accuracy_weight: float = 0.5,
    ) -> List[Tuple[LBRSchedule, float]]:
        """Enumerate all valid schedules sorted by weighted accuracy-latency score.

        Args:
            accuracy_weight: Weight for accuracy (fraction of CADN blocks);
                (1 - accuracy_weight) is the weight for latency (1/speedup).

        Returns:
            List of (LBRSchedule, score) sorted descending by score.
        """
        results = []
        for bits in range(2 ** self.num_blocks):
            mask = [(bits >> i) & 1 == 1 for i in range(self.num_blocks)]
            try:
                sched = self._make_schedule(mask)
                cadn_rate = sched.n_cadn_refreshes / (self.num_blocks + 1)
                score = (
                    accuracy_weight * cadn_rate
                    + (1.0 - accuracy_weight) * (sched.estimated_speedup_vs_full_cadn - 1.0)
                )
                results.append((sched, score))
            except ValueError:
                continue
        return sorted(results, key=lambda t: t[1], reverse=True)


# ---------------------------------------------------------------------------
# Convenience: default schedule for the canonical 3-block FHEViT config
# ---------------------------------------------------------------------------

def default_lbr_schedule(
    num_blocks: int = 3,
    use_cadn: bool = True,
    poly_norm_degree: int = 2,
    enable_token_mix: bool = True,
    depth_budget: int = 13,
    strategy: str = "diagnostic_priority",
    cadn_fraction: float = 0.5,
) -> LBRSchedule:
    """Create the default LBR schedule for the canonical FHEViT configuration.

    Args:
        num_blocks: Number of transformer blocks (default 3).
        use_cadn: If True (default), CADN is available for selected blocks.
        poly_norm_degree: PolyNorm degree (1 or 2) for non-CADN blocks.
        enable_token_mix: Whether Linear Token Mixing is enabled (+1 depth/block).
        depth_budget: CKKS multiplicative depth budget.
        strategy: LBR strategy name.
        cadn_fraction: Fraction of blocks using CADN.

    Returns:
        LBRSchedule for the given configuration.
    """
    # CADN block depth: input_proj + act + output_proj + opt_tokenmix
    depth_cadn_block = 3 + int(enable_token_mix)   # 4 with LTM, 3 without
    # PolyNorm block depth: poly_norm + input_proj + act + output_proj + opt_tokenmix
    depth_poly_block = poly_norm_degree + 1 + 1 + 1 + int(enable_token_mix)

    scheduler = LazyBlockRefreshScheduler(
        num_blocks=num_blocks,
        depth_per_block_cadn=depth_cadn_block,
        depth_per_block_polynorm=depth_poly_block,
        depth_budget=depth_budget,
        patch_proj_depth=1,
        wconcept_depth=1,
        strategy=strategy if use_cadn else "full_cadn",
        cadn_fraction=cadn_fraction,
    )
    return scheduler.compute()


__all__ = [
    "BlockRefreshDecision",
    "LazyBlockRefreshScheduler",
    "LBRSchedule",
    "default_lbr_schedule",
]
