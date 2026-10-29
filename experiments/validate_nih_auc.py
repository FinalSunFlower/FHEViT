"""Native pneumonia log-odds on the locked NIH probe.

Section 2.5 of EVALUATION_PROTOCOL.md fixes the patients before this run.
Oracle PCA, the teacher mean, and the linear bridge are not fit. MedGemma is
not loaded. The COVID final holdout is not read.
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

from experiments.evaluate_diagnostic_prefix import extract  # noqa: E402
from experiments.nih_pneumonia import (  # noqa: E402
    NEGATIVE_LIMIT,
    POSITIVE_LIMIT,
    PROMPT,
    REPLICATION_NEGATIVE,
    REPLICATION_POSITIVE,
)
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402
from experiments.validate_pneumonia_log_odds import NEGATIVE, POSITIVE, next_logits, single_token_id  # noqa: E402
from fpavlm.evaluation import patient_roc_auc_interval  # noqa: E402

SECTIONS = {
    "2.5": {
        "probe": ROOT / "data" / "nih_chestxray14",
        "positive": POSITIVE_LIMIT,
        "negative": NEGATIVE_LIMIT,
        "schema": "nih_pneumonia_auc_v1",
        "scope": "nih_chestxray14_development_probe",
        "filename": "nih_pneumonia_auc_{}.json",
        "width_note": "Fourteen versus thirteen patients can leave a wide interval.",
    },
    "2.6": {
        "probe": ROOT / "data" / "nih_chestxray14_replication",
        "positive": REPLICATION_POSITIVE,
        "negative": REPLICATION_NEGATIVE,
        "schema": "nih_replication_auc_v1",
        "scope": "nih_chestxray14_replication",
        "filename": "nih_replication_auc_{}.json",
        "width_note": "The section 2.5 patients are excluded from this replication.",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["qwen2vl-2b", "internvl2-2b"], required=True)
    parser.add_argument("--section", choices=sorted(SECTIONS), default="2.5")
    args = parser.parse_args()
    section = SECTIONS[args.section]
    torch.set_num_threads(4)
    started = time.perf_counter()
    pool = json.loads((section["probe"] / "metadata.json").read_text(encoding="utf-8"))
    records = pool.get("records") or []
    expected = section["positive"] + section["negative"]
    if pool.get("shortfall") or not pool.get("pool_complete") or len(records) != expected:
        raise ValueError("NIH probe is incomplete; AUC stays uncomputed")
    if any(row.get("download_error") for row in records):
        raise ValueError("a failed NIH download was not replaced, and AUC stays uncomputed")
    labels_present = [row["label_pneumonia"] for row in records]
    if labels_present.count(1) != section["positive"] or labels_present.count(0) != section["negative"]:
        raise ValueError(f"NIH probe counts do not match section {args.section}")
    if args.section == "2.6":
        sealed = json.loads((ROOT / "data" / "nih_chestxray14" / "metadata.json").read_text(encoding="utf-8"))
        overlap = {row["patient_id"] for row in records} & {row["patient_id"] for row in sealed["records"]}
        if overlap or len(sealed["records"]) != POSITIVE_LIMIT + NEGATIVE_LIMIT:
            raise ValueError("replication reuses a section 2.5 patient")
    probe = section["probe"]
    llm = load_backend(args.backend)
    positive_id = single_token_id(llm.tokenizer, POSITIVE)
    negative_id = single_token_id(llm.tokenizer, NEGATIVE)
    reports = []
    for row in records:
        tokens = extract(llm, probe / row["relative_path"], args.backend)
        logits = next_logits(llm, args.backend, tokens, PROMPT)
        log_odds = float(logits[positive_id] - logits[negative_id])
        reports.append({
            "patient_id": row["patient_id"],
            "relative_path": row["relative_path"],
            "finding_labels": row["finding_labels"],
            "label_pneumonia": row["label_pneumonia"],
            "log_odds": log_odds,
            "prefers_yes": log_odds > 0,
        })
        print(json.dumps({
            "patient_id": row["patient_id"], "label_pneumonia": row["label_pneumonia"],
            "log_odds": round(log_odds, 3),
        }), flush=True)
    interval = patient_roc_auc_interval(
        [row["label_pneumonia"] for row in reports],
        [row["log_odds"] for row in reports],
        [row["patient_id"] for row in reports],
    )
    pneumonia = [row for row in reports if row["label_pneumonia"] == 1]
    normal = [row for row in reports if row["label_pneumonia"] == 0]
    result = {
        "schema": section["schema"],
        "protocol_section": args.section,
        "evidence_scope": section["scope"],
        "backend": args.backend,
        "release": "native_injected_only",
        "prompt": PROMPT,
        "pneumonia_patients": len(pneumonia),
        "no_finding_patients": len(normal),
        "pneumonia_prefer_yes": sum(row["prefers_yes"] for row in pneumonia),
        "no_finding_prefer_yes": sum(row["prefers_yes"] for row in normal),
        "auc": interval,
        "native_ceiling_above_chance": interval["above_chance"],
        "yes_no_bias_calibrated": False,
        "constant_shift_checked_by_rescore": False,
        "pca_refit": False,
        "bridge_refit": False,
        "teacher_mean_refit": False,
        "medgemma_ran": False,
        "release_rescored": False,
        "official_test_list_read": False,
        "covid_final_holdout_opened": False,
        "fhe_executed": False,
        "reports": reports,
        "limitations": [
            "NIH No Finding is the released NLP label, not a radiologist normal.",
            "A pneumonia row may carry other NLP tokens.",
            section["width_note"],
            "Oracle PCA, the teacher mean, and the bridge were not refit.",
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    destination = ROOT / "experiments" / "results" / section["filename"].format(args.backend)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "backend": args.backend,
        "auc": interval["auc"],
        "one_sided_95_lower": interval["one_sided_95_lower"],
        "above_chance": interval["above_chance"],
        "pneumonia_prefer_yes": result["pneumonia_prefer_yes"],
        "no_finding_prefer_yes": result["no_finding_prefer_yes"],
    }, indent=2))


if __name__ == "__main__":
    main()
