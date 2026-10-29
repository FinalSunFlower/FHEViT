"""Read the five rank-1999 redraws and the -0.025 confirmation power.

EVALUATION_PROTOCOL.md section 2.33. This file does not score a new cell
and does not download the reserved confirmation images.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.nih_sample_size import power_noninferior, se_at_n  # noqa: E402

REDRAWS = ROOT / "experiments" / "results" / "nih_rank1999_redraws_internvl2-2b.json"
BRIDGE = ROOT / "experiments" / "results" / "nih_unlabeled_bridge_internvl2-2b.json"
DESTINATION = ROOT / "experiments" / "results" / "nih_redraw_reading.json"
RESERVED_PA_POSITIVES = 301


def clopper_pearson(successes, trials, alpha=0.05):
    """Two-sided Clopper-Pearson interval. Five draws are not a new sample."""
    from scipy.stats import beta

    if not 0 <= successes <= trials or trials < 1 or not 0 < alpha < 1:
        raise ValueError("the binomial interval needs a valid count")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def overlap_fractions(id_lists):
    """Share of the first draw's size that also sits in the second draw."""
    fractions = []
    for left in range(len(id_lists)):
        first = set(id_lists[left])
        if not first:
            raise ValueError("a redraw is empty")
        for right in range(left + 1, len(id_lists)):
            fractions.append(len(first & set(id_lists[right])) / len(first))
    return fractions


def main():
    redraws = json.loads(REDRAWS.read_text(encoding="utf-8"))
    cells = redraws["cells"]
    if len(cells) != 5 or not redraws.get("complete"):
        raise SystemExit("the five redraws are not complete")
    differences = [cell["pa_only"]["difference"] for cell in cells]
    lowers = [cell["pa_only"]["one_sided_95_lower"] for cell in cells]
    passes = sum(bool(cell["passes"]) for cell in cells)
    fractions = overlap_fractions([cell["patient_ids"] for cell in cells])
    bridge = json.loads(BRIDGE.read_text(encoding="utf-8"))
    passing = [cell for cell in bridge["cells"] if cell.get("passes")]
    if len(passing) != 1:
        raise SystemExit("the published bridge does not have one passing cell")
    deviation = float(passing[0]["power"]["bootstrap_sd"])
    positives = int(passing[0]["power"]["pa_positives"])
    se = se_at_n(deviation, RESERVED_PA_POSITIVES, positives)
    lower, upper = clopper_pearson(passes, 5)
    payload = {
        "schema": "nih_redraw_reading_v1",
        "protocol_section": "2.33",
        "source": "nih_rank1999_redraws_internvl2-2b.json",
        "difference_mean": statistics.fmean(differences),
        "lower_mean": statistics.fmean(lowers),
        "lower_sd": statistics.stdev(lowers),
        "passes": passes,
        "trials": 5,
        "clopper_pearson_95": [lower, upper],
        "overlap_fraction_min": min(fractions),
        "overlap_fraction_median": statistics.median(fractions),
        "overlap_fraction_max": max(fractions),
        "expected_overlap_fraction": 2000 / 3000,
        "bootstrap_sd": deviation,
        "pa_positives_ref": positives,
        "reserved_pa_positives": RESERVED_PA_POSITIVES,
        "se_at_301": se,
        "power_at_minus_0_02": power_noninferior(se, -0.02),
        "power_at_minus_0_025": power_noninferior(se, -0.025),
        "confirmation_opened": False,
        "reserved_images_downloaded": False,
        "margin_z_from_lower_mean": (-0.05 - statistics.fmean(lowers)) / statistics.stdev(lowers),
        "normal_pass_probability_from_lower_sd": float(
            statistics.NormalDist().cdf(-(-0.05 - statistics.fmean(lowers)) / statistics.stdev(lowers))
        ),
    }
    if not math.isfinite(payload["power_at_minus_0_025"]):
        raise SystemExit("the -0.025 power is not finite")
    DESTINATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "difference_mean": payload["difference_mean"],
        "lower_mean": payload["lower_mean"],
        "lower_sd": payload["lower_sd"],
        "clopper_pearson_95": payload["clopper_pearson_95"],
        "overlap_fraction_median": payload["overlap_fraction_median"],
        "power_at_minus_0_025": payload["power_at_minus_0_025"],
        "confirmation_opened": False,
    }), flush=True)


if __name__ == "__main__":
    main()
