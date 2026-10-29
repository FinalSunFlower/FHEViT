"""Pass probability near a -0.02 gap, before any section 2.27 score.

The standard error is the normal reading of the published one-sided bound
of the closest section 2.25 cell. No image is downloaded and the reserve
is not opened.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.nih_pa_only_power import required_positives  # noqa: E402
from experiments.nih_sample_size import power_noninferior, se_at_n  # noqa: E402

GRID = ROOT / "experiments" / "results" / "nih_prefix_grid_internvl2-2b.json"
PA_POWER = ROOT / "experiments" / "results" / "nih_pa_only_power.json"
DELTAS = (0.0, -0.02)
DESTINATION = ROOT / "experiments" / "results" / "nih_full_rank_power.json"


def se_from_one_sided_bound(difference, lower):
    """Normal standard error implied by a published one-sided 95% bound."""
    gap = difference - lower
    if gap <= 0 or not math.isfinite(gap):
        raise ValueError("the one-sided bound must lie below the point estimate")
    return gap / statistics.NormalDist().inv_cdf(0.95)


def closest_cell(cells):
    if not cells:
        raise ValueError("the prefix grid has no cells")
    best = max(cell["pa_only"]["one_sided_95_lower"] for cell in cells)
    chosen = [cell for cell in cells if cell["pa_only"]["one_sided_95_lower"] == best]
    if len(chosen) != 1:
        raise ValueError("the closest PA-only cell is not unique")
    return chosen[0]


def main():
    grid = json.loads(GRID.read_text(encoding="utf-8"))
    if grid.get("confirmation_scored") or grid.get("reserved_confirmation_used"):
        raise SystemExit("section 2.25 used the reserved confirmation patients")
    cell = closest_cell(grid["cells"])
    pa = cell["pa_only"]
    if pa["patient_count"] != 368 or pa["noninferior"]:
        raise SystemExit("the closest cell is not the failed 368-patient PA comparison")
    prior = json.loads(PA_POWER.read_text(encoding="utf-8"))
    n_pos_ref = int(prior["development_pa_positives"])
    n_reserved = int(prior["reserved_pneumonia_pa"])
    if n_pos_ref != 129 or n_reserved != 301:
        raise SystemExit("the PA-positive counts are not the section 2.25 counts")
    se_ref = se_from_one_sided_bound(pa["difference"], pa["one_sided_95_lower"])
    se_reserved = se_at_n(se_ref, n_reserved, n_pos_ref)
    plans = [
        {
            "true_difference": delta,
            "power_at_reserved_pa": power_noninferior(se_reserved, delta),
            "required_pa_positives": required_positives(se_ref, n_pos_ref, delta),
        }
        for delta in DELTAS
    ]
    near = next(row for row in plans if row["true_difference"] == -0.02)
    result = {
        "schema": "nih_full_rank_power_v1",
        "protocol_section": "2.27",
        "source_cell": {
            "basis": cell["basis"],
            "bridge": cell["bridge"],
            "rank": cell["rank"],
            "difference": pa["difference"],
            "one_sided_95_lower": pa["one_sided_95_lower"],
        },
        "planning_se_on_129_pa_positives": se_ref,
        "planning_se_on_301_pa_positives": se_reserved,
        "plans": plans,
        "power_at_minus_0_02_reaches_0_8": near["power_at_reserved_pa"] >= 0.8,
        "confirmation_opened": False,
        "images_downloaded": False,
        "new_cell_scored": False,
        "limitations": [
            "This standard error is the normal reading of a published bound, not a new bootstrap.",
            "The reserve opens only from a passing cell's own bootstrap standard deviation.",
        ],
    }
    DESTINATION.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "cell": result["source_cell"],
        "se_129": se_ref,
        "se_301": se_reserved,
        "plans": plans,
        "power_at_minus_0_02_reaches_0_8": result["power_at_minus_0_02_reaches_0_8"],
    }, indent=2))


if __name__ == "__main__":
    main()
