"""Pneumonia log-odds AUC once the locked Open-I negatives exist.

Positives stay the 20 development patients. Negatives are the downloaded pool from
section 2.2. The Yes/No preference is reported and is not calibrated. The final
holdout is not read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import PATIENT_ROOT  # noqa: E402
from experiments.download_patient_cohort import validate_manifest  # noqa: E402
from experiments.evaluate_diagnostic_prefix import extract  # noqa: E402
from experiments.openi_negatives import POOL_MINIMUM  # noqa: E402
from experiments.validate_pneumonia_log_odds import (  # noqa: E402
    NEGATIVE,
    POSITIVE,
    PROMPT,
    REMOTE_CODE_SHA256,
    next_logits,
    pneumonia_cases,
    single_token_id,
)
from fpavlm import InternVL2Adapter, Qwen2VLAdapter  # noqa: E402
from fpavlm.checkpoint import load_tensor_checkpoint  # noqa: E402
from fpavlm.evaluation import patient_roc_auc_interval  # noqa: E402

NEGATIVES = ROOT / "data" / "openi_negatives"


def load_backend(backend):
    if backend == "qwen2vl-2b":
        llm = Qwen2VLAdapter(str(ROOT / "models" / backend))
    else:
        llm = InternVL2Adapter(
            str(ROOT / "models" / backend), allow_remote_code=True, remote_code_sha256=REMOTE_CODE_SHA256,
        )
    llm.load()
    return llm


def score(llm, backend, path, positive_id, negative_id):
    tokens = extract(llm, path, backend)
    logits = next_logits(llm, backend, tokens, PROMPT)
    return float(logits[positive_id] - logits[negative_id])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["qwen2vl-2b", "internvl2-2b"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    pool = json.loads((NEGATIVES / "metadata.json").read_text(encoding="utf-8"))
    negatives = [row for row in pool["records"] if not row.get("download_error")]
    if len(negatives) < POOL_MINIMUM:
        raise ValueError(f"negative pool has {len(negatives)} images; AUC stays uncomputed below {POOL_MINIMUM}")
    evidence = json.loads((ROOT / "experiments/results/spatial_quadratic_diagnostic.json").read_text("utf-8"))
    checkpoint = load_tensor_checkpoint(
        str(ROOT / "artifacts/spatial_quadratic_diagnostic.pt"),
        expected_sha256=evidence["checkpoint_sha256"],
    )
    records = json.loads((PATIENT_ROOT / "metadata.json").read_text("utf-8"))
    validate_manifest(records, PATIENT_ROOT)
    rows = records["records"]
    known = set(checkpoint["training_patients"]) | set(checkpoint["evaluation_patients"])
    positives = pneumonia_cases(rows)
    positive_ids = {rows[index]["patient_id"] for index in positives}
    if len(positives) != 20 or not positive_ids <= known:
        raise ValueError(f"expected 20 pneumonia patients inside the frozen split, found {len(positives)}")
    llm = load_backend(args.backend)
    positive_id = single_token_id(llm.tokenizer, POSITIVE)
    negative_id = single_token_id(llm.tokenizer, NEGATIVE)
    reports = []
    for index in positives:
        row = rows[index]
        log_odds = score(llm, args.backend, PATIENT_ROOT / row["relative_path"], positive_id, negative_id)
        reports.append({
            "patient_id": f"dev:{row['patient_id']}", "source": "development",
            "relative_path": row["relative_path"], "finding": row.get("finding", ""),
            "label_pneumonia": 1, "log_odds": log_odds, "prefers_yes": log_odds > 0,
        })
        print(json.dumps({"patient_id": reports[-1]["patient_id"], "log_odds": round(log_odds, 3)}), flush=True)
    for row in negatives:
        log_odds = score(
            llm, args.backend, NEGATIVES / row["relative_path"], positive_id, negative_id,
        )
        reports.append({
            "patient_id": f"openi:{row['uid']}", "source": "openi_negative_pool",
            "relative_path": row["relative_path"], "finding": row.get("impression", ""),
            "label_pneumonia": 0, "log_odds": log_odds, "prefers_yes": log_odds > 0,
        })
        print(json.dumps({"patient_id": reports[-1]["patient_id"], "log_odds": round(log_odds, 3)}), flush=True)
    labels = [row["label_pneumonia"] for row in reports]
    scores = [row["log_odds"] for row in reports]
    patients = [row["patient_id"] for row in reports]
    interval = patient_roc_auc_interval(labels, scores, patients)
    labeled = [row for row in reports if row["label_pneumonia"] == 1]
    normal = [row for row in reports if row["label_pneumonia"] == 0]
    result = {
        "schema": "pneumonia_auc_v1", "evidence_scope": "development_positives_plus_locked_openi_negatives",
        "backend": args.backend, "release": "native_injected_only", "prompt": PROMPT,
        "positive_patients": len(labeled), "negative_patients": len(normal),
        "positives_prefer_yes": sum(row["prefers_yes"] for row in labeled),
        "negatives_prefer_yes": sum(row["prefers_yes"] for row in normal),
        "auc": interval, "native_ceiling_above_chance": interval["above_chance"],
        "yes_no_bias_calibrated": False, "final_holdout_opened": False,
        "medgemma_ran": False, "release_rescored": False, "attacks_ran": False,
        "fhe_executed": False, "reports": reports,
        "limitations": [
            "Negatives are Open-I reports under the locked normal allow-list, not the COVID cohort.",
            "A Yes preference on both classes can still leave the ranking AUC above 0.5.",
            "The final holdout was not opened.",
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    destination = ROOT / "experiments/results" / f"pneumonia_auc_{args.backend}.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "backend": args.backend, "auc": interval["auc"],
        "one_sided_95_lower": interval["one_sided_95_lower"],
        "above_chance": interval["above_chance"],
        "positives_prefer_yes": result["positives_prefer_yes"],
        "negatives_prefer_yes": result["negatives_prefer_yes"],
    }, indent=2))


if __name__ == "__main__":
    main()
