"""Does the true oracle PCA bridge pass AUC non-inferiority on NIH?

EVALUATION_PROTOCOL.md section 2.16. The codes are the development PCA
coordinates, not a predictor. InternVL2 only. No multilayer perceptron.
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
from experiments.fit_nih_oracle_bridge import (  # noqa: E402
    CONFIRMATION,
    DEVELOPMENT,
    RANKS,
    _native,
    _records,
)
from experiments.nih_pneumonia import (  # noqa: E402
    CONFIRMATION_NEGATIVE,
    CONFIRMATION_POSITIVE,
    PROMPT,
    REPLICATION_NEGATIVE,
    REPLICATION_POSITIVE,
)
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402
from experiments.validate_pneumonia_log_odds import NEGATIVE, POSITIVE, next_logits, single_token_id  # noqa: E402
from fpavlm import DiagnosticPrefixProjector, PrefixSpec  # noqa: E402
from fpavlm.evaluation import paired_auc_noninferiority  # noqa: E402

BACKEND = "internvl2-2b"
WIDTH = ROOT / "experiments" / "results" / "nih_oracle_bridge_internvl2-2b.json"
NATIVE = ROOT / "experiments" / "results" / "nih_finite_state_auc_internvl2-2b.json"


def coordinate_path_remains_open(noninferior):
    """One passing rank keeps the coordinate path open. An empty list does not."""
    if not noninferior:
        raise ValueError("every locked rank needs a non-inferiority bit")
    return any(noninferior)


def _log_odds(llm, tokens, positive_id, negative_id):
    logits = next_logits(llm, BACKEND, tokens, PROMPT)
    return float(logits[positive_id] - logits[negative_id])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=[BACKEND], default=BACKEND)
    args = parser.parse_args()
    if args.backend != BACKEND:
        raise SystemExit("section 2.16 fits internvl2-2b only; Qwen stays the secondary line")
    torch.set_num_threads(4)
    torch.manual_seed(812)
    started = time.perf_counter()
    stored = json.loads(WIDTH.read_text(encoding="utf-8"))
    published = json.loads(NATIVE.read_text(encoding="utf-8"))
    stored_mse = {
        row["release"]: row["mse_improvement_over_mean"]
        for row in stored["releases"] if row["release"].startswith("pca_rank_")
    }
    if list(stored_mse) != [f"pca_rank_{rank}" for rank in RANKS]:
        raise SystemExit("section 2.7 does not have the locked oracle ranks")
    development = _records(DEVELOPMENT, REPLICATION_POSITIVE + REPLICATION_NEGATIVE)
    confirmation = _records(CONFIRMATION, CONFIRMATION_POSITIVE + CONFIRMATION_NEGATIVE)
    dev_ids = [row["patient_id"] for row in development]
    conf_ids = [row["patient_id"] for row in confirmation]
    labels = [row["label_pneumonia"] for row in confirmation]
    if set(dev_ids) & set(conf_ids):
        raise ValueError("development and confirmation patients overlap")
    llm = load_backend(BACKEND)
    native = torch.cat([
        _native(llm, DEVELOPMENT, development, BACKEND),
        _native(llm, CONFIRMATION, confirmation, BACKEND),
    ]).float()
    train_n = len(development)
    test = native[train_n:]
    train_mean = native[:train_n].mean(0, keepdim=True)
    centered = native.reshape(len(native), -1).double() - native[:train_n].reshape(train_n, -1).double().mean(0)
    basis = torch.linalg.svd(centered[:train_n], full_matrices=False).Vh.T
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
            _fingerprint({"release": "nih_pca_oracle_auc", "rank": rank, "section": "2.16"}),
        )
        fitted = DiagnosticPrefixProjector(spec).fit(code[:train_n], native[:train_n], sigma=0.)
        prediction = fitted(code[train_n:])
        width = summarize(f"pca_rank_{rank}", prediction, test, train_mean, conf_ids, bootstrap_samples=2000)
        if abs(width["mse_improvement_over_mean"] - stored_mse[f"pca_rank_{rank}"]) > 0.001:
            raise SystemExit(f"rank {rank} MSE disagrees with section 2.7")
        release_scores = [
            _log_odds(llm, prediction[offset].unsqueeze(0), positive_id, negative_id)
            for offset in range(len(confirmation))
        ]
        comparison = paired_auc_noninferiority(labels, native_scores, release_scores, conf_ids)
        if abs(comparison["auc_native"] - published["auc"]["auc_native"]) > 1e-4:
            raise SystemExit("native AUC disagrees with section 2.7")
        rows.append({
            "rank": rank,
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
    remains = coordinate_path_remains_open([row["noninferior"] for row in rows])
    result = {
        "schema": "nih_oracle_auc_v1",
        "protocol_section": "2.16",
        "backend": BACKEND,
        "qwen_fitted": False,
        "qwen_secondary_line": True,
        "development_patients": train_n,
        "confirmation_patients": len(confirmation),
        "same_confirmation_patients": True,
        "fit_saw_confirmation": False,
        "prompt": PROMPT,
        "ranks": rows,
        "coordinate_path_remains_open": remains,
        "mlp_started": False,
        "clinical_label_training_started": False,
        "rbf_grid_extended": False,
        "regional_release_started": False,
        "official_test_list_read": False,
        "limitations": [
            "The release is the true PCA code, not a predictor and not an encrypted symbol.",
            "Non-inferiority is the pneumonia log-odds AUC, not token MSE.",
            "A multilayer perceptron and clinical-label training are not started from this result.",
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    destination = ROOT / "experiments" / "results" / "nih_oracle_auc_internvl2-2b.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"coordinate_path_remains_open": remains}, indent=2))


if __name__ == "__main__":
    main()
