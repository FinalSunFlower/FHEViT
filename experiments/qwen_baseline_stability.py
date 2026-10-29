"""How far does Qwen's native AUC move across 72-patient redraws?

EVALUATION_PROTOCOL.md section 2.9. Development log-odds stay as stored.
Confirmation log-odds are scored only when the per-patient rows are missing,
and only if that AUC matches the section 2.7 native AUC. No new slice is drawn.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.nih_pneumonia import PROMPT  # noqa: E402
from fpavlm.evaluation import patient_roc_auc_interval, roc_auc  # noqa: E402

DRAWS = 200
SEED = 42
TAKE = 36
DEVELOPMENT = ROOT / "experiments" / "results" / "nih_replication_auc_qwen2vl-2b.json"
PUBLISHED = ROOT / "experiments" / "results" / "nih_finite_state_auc_qwen2vl-2b.json"
CONFIRMATION = ROOT / "data" / "nih_chestxray14_confirmation"


def balanced_redraws(labels, scores, *, draws=DRAWS, seed=SEED, take=TAKE):
    """Point AUCs of balanced patient subsets. The seed fixes the redraws."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.shape != y.shape or not np.isfinite(s).all():
        raise ValueError("labels and finite scores must share one shape")
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if len(positive) < take or len(negative) < take or draws < 1:
        raise ValueError("each class must cover the redraw, and at least one draw is required")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        chosen = np.concatenate([
            rng.choice(positive, size=take, replace=False),
            rng.choice(negative, size=take, replace=False),
        ])
        value = roc_auc(y[chosen], s[chosen])
        if value is None:
            raise ValueError("a balanced redraw is missing a class")
        values.append(value)
    samples = np.asarray(values, dtype=float)
    low = float(np.quantile(samples, 0.025))
    high = float(np.quantile(samples, 0.975))
    return {
        "draws": draws,
        "seed": seed,
        "take_per_class": take,
        "auc_mean": float(samples.mean()),
        "auc_std": float(samples.std()),
        "auc_min": float(samples.min()),
        "auc_max": float(samples.max()),
        "percentile_2_5": low,
        "percentile_97_5": high,
        "both_above_0_5": bool(low > 0.5 and high > 0.5),
        "aucs": [float(value) for value in samples],
    }


def _rows(payload):
    rows = payload["reports"]
    return (
        [row["label_pneumonia"] for row in rows],
        [row["log_odds"] for row in rows],
        [row["patient_id"] for row in rows],
    )


def _score_confirmation():
    import torch

    from experiments.evaluate_diagnostic_prefix import extract
    from experiments.validate_pneumonia_auc import load_backend
    from experiments.validate_pneumonia_log_odds import NEGATIVE, POSITIVE, next_logits, single_token_id

    torch.set_num_threads(4)
    manifest = json.loads((CONFIRMATION / "metadata.json").read_text(encoding="utf-8"))
    records = manifest["records"]
    llm = load_backend("qwen2vl-2b")
    positive_id = single_token_id(llm.tokenizer, POSITIVE)
    negative_id = single_token_id(llm.tokenizer, NEGATIVE)
    reports = []
    for row in records:
        tokens = extract(llm, CONFIRMATION / row["relative_path"], "qwen2vl-2b")
        logits = next_logits(llm, "qwen2vl-2b", tokens, PROMPT)
        log_odds = float(logits[positive_id] - logits[negative_id])
        reports.append({
            "patient_id": row["patient_id"],
            "label_pneumonia": row["label_pneumonia"],
            "log_odds": log_odds,
        })
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    started = time.perf_counter()
    development = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    published_auc = float(published["auc"]["auc_native"])
    dev_y, dev_s, dev_ids = _rows(development)
    reports = _score_confirmation()
    conf_y = [row["label_pneumonia"] for row in reports]
    conf_s = [row["log_odds"] for row in reports]
    conf_ids = [row["patient_id"] for row in reports]
    if set(dev_ids) & set(conf_ids):
        raise ValueError("development and confirmation patients overlap")
    if conf_y.count(1) != TAKE or conf_y.count(0) != TAKE:
        raise ValueError("confirmation is not 36 and 36")
    if dev_y.count(1) != TAKE or dev_y.count(0) != TAKE:
        raise ValueError("development is not 36 and 36")
    recomputed = roc_auc(conf_y, conf_s)
    if recomputed is None or abs(recomputed - published_auc) > 1e-4:
        raise SystemExit(
            f"recomputed confirmation AUC {recomputed} disagrees with the section 2.7 native AUC")
    confirmation_interval = patient_roc_auc_interval(conf_y, conf_s, conf_ids)
    redraw = balanced_redraws(dev_y + conf_y, dev_s + conf_s)
    stays = bool(redraw["both_above_0_5"])
    result = {
        "schema": "qwen_native_stability_v1",
        "protocol_section": "2.9",
        "backend": "qwen2vl-2b",
        "prompt": PROMPT,
        "development_source": "experiments/results/nih_replication_auc_qwen2vl-2b.json",
        "development_auc": development["auc"]["auc"],
        "confirmation_auc": recomputed,
        "confirmation_matches_section_2_7": True,
        "confirmation_interval": confirmation_interval,
        "redraw": {key: value for key, value in redraw.items() if key != "aucs"},
        "redraw_aucs": redraw["aucs"],
        "qwen_stays_release_backend": stays,
        "prompt_changed": False,
        "scores_calibrated": False,
        "new_slice_downloaded": False,
        "medgemma_ran": False,
        "official_test_list_read": False,
        "elapsed_s": time.perf_counter() - started,
    }
    destination = ROOT / "experiments" / "results" / "qwen_native_stability.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "confirmation_auc": recomputed,
        "confirmation_one_sided_95_lower": confirmation_interval["one_sided_95_lower"],
        "percentile_2_5": redraw["percentile_2_5"],
        "percentile_97_5": redraw["percentile_97_5"],
        "qwen_stays_release_backend": stays,
    }, indent=2))


if __name__ == "__main__":
    main()
