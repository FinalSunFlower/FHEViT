"""Refit the InternVL2 oracle PCA bridge on the NIH development patients.

EVALUATION_PROTOCOL.md section 2.7. The basis and the bridge see development
images only. Width is scored on the confirmation patients. Qwen waits until
this InternVL2 result is on disk. The 0.4 release is not scored here.
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
from experiments.evaluate_diagnostic_prefix import extract  # noqa: E402
from experiments.nih_pneumonia import (  # noqa: E402
    CONFIRMATION_NEGATIVE,
    CONFIRMATION_POSITIVE,
    REPLICATION_NEGATIVE,
    REPLICATION_POSITIVE,
)
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402
from fpavlm import DiagnosticPrefixProjector, PrefixSpec  # noqa: E402

DEVELOPMENT = ROOT / "data" / "nih_chestxray14_replication"
CONFIRMATION = ROOT / "data" / "nih_chestxray14_confirmation"
RANKS = (4, 16, 64)
PRIOR = ROOT / "experiments" / "results" / "nih_oracle_bridge_internvl2-2b.json"


def _records(path, expected):
    manifest = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    records = manifest.get("records") or []
    if not manifest.get("pool_complete") or len(records) != expected:
        raise ValueError(f"{path.name} is not a complete split")
    if any(row.get("download_error") for row in records):
        raise ValueError(f"{path.name} has a failed download")
    return records


def _native(llm, root, records, backend):
    return torch.cat([extract(llm, root / row["relative_path"], backend) for row in records]).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["internvl2-2b", "qwen2vl-2b"], default="internvl2-2b")
    args = parser.parse_args()
    if args.backend == "qwen2vl-2b" and not PRIOR.is_file():
        raise SystemExit("InternVL2 width result is not on disk; Qwen stays unfitted")
    torch.set_num_threads(4)
    torch.manual_seed(812)
    started = time.perf_counter()
    development = _records(DEVELOPMENT, REPLICATION_POSITIVE + REPLICATION_NEGATIVE)
    confirmation = _records(CONFIRMATION, CONFIRMATION_POSITIVE + CONFIRMATION_NEGATIVE)
    dev_ids = [row["patient_id"] for row in development]
    conf_ids = [row["patient_id"] for row in confirmation]
    if set(dev_ids) & set(conf_ids):
        raise ValueError("development and confirmation patients overlap")
    llm = load_backend(args.backend)
    native = torch.cat([
        _native(llm, DEVELOPMENT, development, args.backend),
        _native(llm, CONFIRMATION, confirmation, args.backend),
    ]).float()
    train_n = len(development)
    train_mean = native[:train_n].mean(0, keepdim=True)
    native_flat = native.reshape(len(native), -1).double()
    centered = native_flat - native_flat[:train_n].mean(0, keepdim=True)
    basis = torch.linalg.svd(centered[:train_n], full_matrices=False).Vh.T
    digest = hashlib.sha256("\n".join(dev_ids).encode("utf-8")).hexdigest()
    hidden = int(native.shape[-1])
    tokens = int(native.shape[-2])
    summaries = [summarize(
        "constant_mean", train_mean.expand(len(confirmation), -1, -1),
        native[train_n:], train_mean, conf_ids, bootstrap_samples=2000,
    )]
    for rank in RANKS:
        code = (centered @ basis[:, :rank]).reshape(len(native), rank, 1).float()
        spec = PrefixSpec(
            args.backend, (rank, 1), tokens, hidden, digest,
            _fingerprint({"release": "nih_pca_oracle", "rank": rank, "section": "2.7"}),
        )
        fitted = DiagnosticPrefixProjector(spec).fit(code[:train_n], native[:train_n], sigma=0.)
        prediction = fitted(code[train_n:])
        summaries.append(summarize(
            f"pca_rank_{rank}", prediction, native[train_n:], train_mean, conf_ids, bootstrap_samples=2000,
        ))
    oracle_beats = [row["release"] for row in summaries if row["release"].startswith("pca_rank_")
                    and row["beats_mean_significantly"]]
    result = {
        "schema": "nih_oracle_bridge_v1",
        "protocol_section": "2.7",
        "backend": args.backend,
        "development_patients": len(development),
        "confirmation_patients": len(confirmation),
        "patient_disjoint": True,
        "fit_saw_confirmation": False,
        "ranks": list(RANKS),
        "native_tokens": tokens,
        "native_hidden": hidden,
        "development_id_sha256": digest,
        "releases": summaries,
        "oracle_ranks_beating_mean": oracle_beats,
        "finite_state_release_scored": False,
        "qwen_fitted": args.backend == "qwen2vl-2b",
        "medgemma_ran": False,
        "auc_noninferiority_computed": False,
        "official_test_list_read": False,
        "limitations": [
            "Width is token MSE on the confirmation patients, not the pneumonia AUC.",
            "PCA ranks are an oracle upper bound, not an encrypted release.",
            "The 0.4 finite-state release stays unscored until an oracle rank beats the mean.",
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    destination = ROOT / "experiments" / "results" / f"nih_oracle_bridge_{args.backend}.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "backend": args.backend,
        "oracle_ranks_beating_mean": oracle_beats,
        "releases": [{
            "release": row["release"],
            "mse_improvement_over_mean": row["mse_improvement_over_mean"],
            "beats_mean_significantly": row["beats_mean_significantly"],
        } for row in summaries],
    }, indent=2))


if __name__ == "__main__":
    main()
