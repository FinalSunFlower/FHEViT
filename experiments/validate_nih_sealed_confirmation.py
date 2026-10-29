"""Score oracle ranks 4 and 8 on the sealed image-level confirmation set.

EVALUATION_PROTOCOL.md section 2.22. The bridge sees the existing development
patients only. Rank 4 is primary. Rank 8 is the secondary comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.diagnose_release_sufficiency import _fingerprint, summarize  # noqa: E402
from experiments.fit_nih_oracle_bridge import DEVELOPMENT, _native, _records  # noqa: E402
from experiments.nih_image_level_census import PER_CLASS  # noqa: E402
from experiments.nih_pneumonia import (  # noqa: E402
    PROMPT,
    REPLICATION_NEGATIVE,
    REPLICATION_POSITIVE,
)
from experiments.validate_nih_oracle_auc import _log_odds  # noqa: E402
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402
from experiments.validate_pneumonia_log_odds import NEGATIVE, POSITIVE, single_token_id  # noqa: E402
from fpavlm import DiagnosticPrefixProjector, PrefixSpec  # noqa: E402
from fpavlm.evaluation import paired_auc_noninferiority  # noqa: E402

BACKEND = "internvl2-2b"
RANKS = (4, 8)
SEALED = ROOT / "data" / "nih_chestxray14_sealed_confirmation"
OLD = (
    ROOT / "data" / "nih_chestxray14" / "metadata.json",
    ROOT / "data" / "nih_chestxray14_replication" / "metadata.json",
    ROOT / "data" / "nih_chestxray14_confirmation" / "metadata.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=[BACKEND], default=BACKEND)
    args = parser.parse_args()
    if args.backend != BACKEND:
        raise SystemExit("section 2.22 scores internvl2-2b only")
    manifest = json.loads((SEALED / "metadata.json").read_text(encoding="utf-8"))
    if not manifest.get("pool_complete") or manifest.get("primary_rank") != 4 or manifest.get("secondary_rank") != 8:
        raise SystemExit("the sealed images are not complete, or the locked ranks changed")
    confirmation = manifest["records"]
    if len(confirmation) != 2 * PER_CLASS or any(row.get("download_error") for row in confirmation):
        raise SystemExit("scoring requires every sealed image")
    if sum(row["label_pneumonia"] == 1 for row in confirmation) != PER_CLASS:
        raise SystemExit("the sealed confirmation set is not balanced")
    old_ids = set()
    for path in OLD:
        old_ids.update(str(row["patient_id"]) for row in json.loads(path.read_text(encoding="utf-8"))["records"])
    if set(row["patient_id"] for row in confirmation) & old_ids:
        raise SystemExit("a previously sealed patient entered the new confirmation set")
    torch.set_num_threads(4)
    torch.manual_seed(812)
    started = time.perf_counter()
    development = _records(DEVELOPMENT, REPLICATION_POSITIVE + REPLICATION_NEGATIVE)
    if set(row["patient_id"] for row in development) & set(row["patient_id"] for row in confirmation):
        raise SystemExit("development and the sealed confirmation overlap")
    llm = load_backend(BACKEND)
    native = torch.cat([
        _native(llm, DEVELOPMENT, development, BACKEND),
        _native(llm, SEALED, confirmation, BACKEND),
    ]).float()
    train_n = len(development)
    test = native[train_n:]
    train_mean = native[:train_n].mean(0, keepdim=True)
    centered = native.reshape(len(native), -1).double() - native[:train_n].reshape(train_n, -1).double().mean(0)
    basis = torch.linalg.svd(centered[:train_n], full_matrices=False).Vh.T
    dev_ids = [row["patient_id"] for row in development]
    conf_ids = [row["patient_id"] for row in confirmation]
    labels = [row["label_pneumonia"] for row in confirmation]
    digest = hashlib.sha256("\n".join(dev_ids).encode("utf-8")).hexdigest()
    tokens, hidden = int(native.shape[-2]), int(native.shape[-1])
    positive_id = single_token_id(llm.tokenizer, POSITIVE)
    negative_id = single_token_id(llm.tokenizer, NEGATIVE)
    native_scores = [
        _log_odds(llm, test[offset].unsqueeze(0), positive_id, negative_id)
        for offset in range(len(confirmation))
    ]
    rows = []
    for rank in RANKS:
        code = (centered @ basis[:, :rank]).reshape(len(native), rank, 1).float()
        spec = PrefixSpec(
            BACKEND, (rank, 1), tokens, hidden, digest,
            _fingerprint({"release": "nih_sealed_pca_oracle", "rank": rank, "section": "2.22"}),
        )
        fitted = DiagnosticPrefixProjector(spec).fit(code[:train_n], native[:train_n], sigma=0.)
        prediction = fitted(code[train_n:])
        width = summarize(f"pca_rank_{rank}", prediction, test, train_mean, conf_ids, bootstrap_samples=2000)
        release_scores = [
            _log_odds(llm, prediction[offset].unsqueeze(0), positive_id, negative_id)
            for offset in range(len(confirmation))
        ]
        comparison = paired_auc_noninferiority(labels, native_scores, release_scores, conf_ids)
        rows.append({
            "rank": rank,
            "role": "primary" if rank == 4 else "secondary",
            "mse_improvement_over_mean": width["mse_improvement_over_mean"],
            "auc": comparison,
            "noninferior": comparison["noninferior"],
            "reports": [
                {
                    "patient_id": conf_ids[offset],
                    "label_pneumonia": labels[offset],
                    "native_log_odds": native_scores[offset],
                    "oracle_log_odds": release_scores[offset],
                }
                for offset in range(len(confirmation))
            ],
        })
        print(json.dumps({
            "rank": rank,
            "auc_native": comparison["auc_native"],
            "auc_oracle": comparison["auc_release"],
            "one_sided_95_lower": comparison["one_sided_95_lower"],
            "noninferior": comparison["noninferior"],
        }), flush=True)
    result = {
        "schema": "nih_sealed_confirmation_auc_v1",
        "protocol_section": "2.22",
        "backend": BACKEND,
        "development_patients": train_n,
        "confirmation_patients": len(confirmation),
        "fit_saw_confirmation": False,
        "prompt": PROMPT,
        "margin": 0.05,
        "ranks": rows,
        "primary_rank": 4,
        "secondary_rank": 8,
        "qwen_fitted": False,
        "predictor_retarget_started": False,
        "mlp_started": False,
        "privacy_channel_redesign_started": False,
        "regional_release_started": False,
        "medgemma_ran": False,
        "direction_nulling_started": False,
        "reconstruction_attack_started": False,
        "official_test_list_read": False,
        "limitations": [
            "The release is the true development PCA code, not a predictor or an encrypted symbol.",
            "Rank 8 is reported and does not replace rank 4.",
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    destination = ROOT / "experiments" / "results" / "nih_sealed_confirmation_auc_internvl2-2b.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest["auc_computed"] = True
    (SEALED / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "primary_noninferior": rows[0]["noninferior"],
        "secondary_noninferior": rows[1]["noninferior"],
    }, indent=2))


if __name__ == "__main__":
    main()
