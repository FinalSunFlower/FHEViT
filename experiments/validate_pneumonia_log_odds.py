"""Forced-choice pneumonia log-odds for one native backbone.

The score is the next-token logit of Yes minus the logit of No, on one
pneumonia-labeled development image per patient. No release is fitted. An AUC is
stored only when both classes exist. This cohort has no confirmed negative, so the
AUC field stays null. The final holdout is not read.
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
from fpavlm import InternVL2Adapter, Qwen2VLAdapter  # noqa: E402
from fpavlm.checkpoint import load_tensor_checkpoint  # noqa: E402
from fpavlm.evaluation import patient_mean_interval, roc_auc  # noqa: E402

PROMPT = "Does this chest X-ray show pneumonia? Answer Yes or No."
POSITIVE = "Yes"
NEGATIVE = "No"
REMOTE_CODE_SHA256 = "94a1159158b0ebef27eb2a5129f21d3c0de89bf9de757684c247546c58baf37e"
MARGIN = 0.05


def single_token_id(tokenizer, text):
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(f"{text!r} is not a single token")
    return int(token_ids[0])


def pneumonia_cases(rows):
    chosen, seen = [], set()
    for index, row in enumerate(rows):
        patient = row["patient_id"]
        if patient in seen or "pneumonia" not in row.get("finding", "").lower():
            continue
        seen.add(patient)
        chosen.append(index)
    return chosen


def unlabeled_cases(rows):
    labeled = {rows[index]["patient_id"] for index in pneumonia_cases(rows)}
    chosen, seen = [], set()
    for index, row in enumerate(rows):
        patient = row["patient_id"]
        if patient in seen or patient in labeled:
            continue
        seen.add(patient)
        chosen.append(index)
    return chosen


def next_logits(llm, backend, tokens, prompt):
    if backend == "qwen2vl-2b":
        _, mask, embeds = llm._build_inputs_embeds(tokens, prompt)
        with torch.no_grad():
            output = llm.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False)
        return output.logits[0, -1].float().cpu()
    token_ids, mask = llm._build_input_ids(tokens, prompt)
    features = tokens.to("cuda", dtype=torch.bfloat16)
    eos_id = llm.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_id is None or eos_id == llm.tokenizer.unk_token_id:
        eos_id = llm.tokenizer.eos_token_id
    with torch.no_grad():
        output = llm.model.generate(
            pixel_values=torch.zeros(1, device="cuda", dtype=torch.bfloat16),
            visual_features=features,
            input_ids=token_ids,
            attention_mask=mask,
            max_new_tokens=1,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=llm.tokenizer.pad_token_id or eos_id,
            eos_token_id=eos_id,
        )
    return output.scores[0][0].float().cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["qwen2vl-2b", "internvl2-2b"], required=True)
    parser.add_argument("--unlabeled", action="store_true",
                        help="score todo patients as an unlabeled control; do not call them negatives")
    args = parser.parse_args()
    if args.unlabeled:
        score_unlabeled(args)
        return
    torch.set_num_threads(4)
    started = time.perf_counter()
    evidence = json.loads((ROOT / "experiments/results/spatial_quadratic_diagnostic.json").read_text("utf-8"))
    checkpoint = load_tensor_checkpoint(
        str(ROOT / "artifacts/spatial_quadratic_diagnostic.pt"),
        expected_sha256=evidence["checkpoint_sha256"],
    )
    records = json.loads((PATIENT_ROOT / "metadata.json").read_text("utf-8"))
    validate_manifest(records, PATIENT_ROOT)
    rows = records["records"]
    evaluation = set(checkpoint["evaluation_patients"])
    cases = pneumonia_cases(rows)
    if len(cases) != 20:
        raise ValueError(f"expected 20 pneumonia patients, found {len(cases)}")
    if args.backend == "qwen2vl-2b":
        llm = Qwen2VLAdapter(str(ROOT / "models" / args.backend))
    else:
        llm = InternVL2Adapter(
            str(ROOT / "models" / args.backend),
            allow_remote_code=True,
            remote_code_sha256=REMOTE_CODE_SHA256,
        )
    llm.load()
    positive_id = single_token_id(llm.tokenizer, POSITIVE)
    negative_id = single_token_id(llm.tokenizer, NEGATIVE)
    reports = []
    for index in cases:
        row = rows[index]
        tokens = extract(llm, PATIENT_ROOT / row["relative_path"], args.backend)
        logits = next_logits(llm, args.backend, tokens, PROMPT)
        log_odds = float(logits[positive_id] - logits[negative_id])
        top = int(logits.argmax())
        entry = {
            "patient_id": row["patient_id"], "relative_path": row["relative_path"],
            "finding": row.get("finding", ""), "evaluation_patient": row["patient_id"] in evaluation,
            "label_pneumonia": 1, "log_odds": log_odds, "prefers_yes": log_odds > 0,
            "top_token": llm.tokenizer.decode([top]),
        }
        reports.append(entry)
        print(json.dumps({
            "patient_id": row["patient_id"], "log_odds": round(log_odds, 3),
            "prefers_yes": entry["prefers_yes"], "top_token": entry["top_token"],
        }), flush=True)
    scores = [row["log_odds"] for row in reports]
    labels = [row["label_pneumonia"] for row in reports]
    patients = [row["patient_id"] for row in reports]
    auc = roc_auc(labels, scores)
    interval = patient_mean_interval(scores, patients)
    prefers = sum(row["prefers_yes"] for row in reports)
    result = {
        "schema": "pneumonia_log_odds_v1", "evidence_scope": "development_only",
        "backend": args.backend, "release": "native_injected_only", "quantization": None,
        "prompt": PROMPT, "positive_token": POSITIVE, "negative_token": NEGATIVE,
        "positive_token_id": positive_id, "negative_token_id": negative_id,
        "patients": len(reports), "positive_patients": len(reports), "negative_patients": 0,
        "prefers_yes": prefers, "log_odds_mean": interval,
        "auc": auc, "auc_defined": auc is not None,
        "chance_line": 0.5, "native_ceiling_above_chance": None,
        "release_margin": MARGIN, "release_noninferiority_evaluable": False,
        "medgemma_ran": False, "release_rescored": False, "attacks_ran": False,
        "final_holdout_opened": False, "fhe_executed": False, "reports": reports,
        "reason_auc_undefined": "every labeled development patient is pneumonia; todo is not a negative",
        "limitations": [
            "Log-odds above zero is a preference for the Yes token, not an AUC.",
            "No confirmed pneumonia-negative patient is in this score.",
            "The final holdout was not opened.",
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    destination = ROOT / "experiments/results" / f"pneumonia_log_odds_{args.backend}.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "backend": args.backend, "prefers_yes": prefers, "patients": len(reports),
        "log_odds_mean": interval["mean"], "ci95": [interval["ci95_lower"], interval["ci95_upper"]],
        "auc": auc,
    }, indent=2))


def score_unlabeled(args):
    """Todo studies are not pneumonia-negatives. This control does not create an AUC."""
    destination = ROOT / "experiments/results" / f"pneumonia_log_odds_{args.backend}.json"
    result = json.loads(destination.read_text(encoding="utf-8"))
    records = json.loads((PATIENT_ROOT / "metadata.json").read_text("utf-8"))
    validate_manifest(records, PATIENT_ROOT)
    rows = records["records"]
    cases = unlabeled_cases(rows)
    if len(cases) != 7:
        raise ValueError(f"expected 7 unlabeled patients, found {len(cases)}")
    if args.backend == "qwen2vl-2b":
        llm = Qwen2VLAdapter(str(ROOT / "models" / args.backend))
    else:
        llm = InternVL2Adapter(
            str(ROOT / "models" / args.backend),
            allow_remote_code=True,
            remote_code_sha256=REMOTE_CODE_SHA256,
        )
    llm.load()
    positive_id = single_token_id(llm.tokenizer, POSITIVE)
    negative_id = single_token_id(llm.tokenizer, NEGATIVE)
    scored = []
    for index in cases:
        row = rows[index]
        tokens = extract(llm, PATIENT_ROOT / row["relative_path"], args.backend)
        logits = next_logits(llm, args.backend, tokens, PROMPT)
        log_odds = float(logits[positive_id] - logits[negative_id])
        scored.append({
            "patient_id": row["patient_id"], "relative_path": row["relative_path"],
            "finding": row.get("finding", ""), "log_odds": log_odds, "prefers_yes": log_odds > 0,
            "top_token": llm.tokenizer.decode([int(logits.argmax())]),
        })
        print(json.dumps(scored[-1] | {"log_odds": round(log_odds, 3)}, default=str), flush=True)
    result["unlabeled_control"] = {
        "patients": len(scored), "prefers_yes": sum(row["prefers_yes"] for row in scored),
        "used_as_negative_class": False, "reports": scored,
    }
    result["auc"] = None
    result["auc_defined"] = False
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "backend": args.backend, "unlabeled_prefers_yes": result["unlabeled_control"]["prefers_yes"],
        "unlabeled_patients": len(scored), "auc": None,
    }, indent=2))


if __name__ == "__main__":
    main()
