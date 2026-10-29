"""Can InternVL2's pooled visual tokens predict its own PCA-4 coordinates?

EVALUATION_PROTOCOL.md section 2.10. The features are the mean of the 256
frozen native tokens. The unpooled matrix is not the predictor. The basis and
every ridge see development patients only. No adapter is trained.
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
from experiments.fit_nih_oracle_bridge import CONFIRMATION, DEVELOPMENT, _native, _records  # noqa: E402
from experiments.nih_pneumonia import (  # noqa: E402
    CONFIRMATION_NEGATIVE,
    CONFIRMATION_POSITIVE,
    REPLICATION_NEGATIVE,
    REPLICATION_POSITIVE,
)
from experiments.validate_nih_predictor_capacity import PATIENT_COUNTS  # noqa: E402
from experiments.validate_pneumonia_auc import load_backend  # noqa: E402
from experiments.validate_pretrained_anchor import (  # noqa: E402
    ALPHAS,
    RELEASE_RANK,
    RIDGE,
    fit_prefix,
    plot_curve,
)
from fpavlm import PrefixSpec  # noqa: E402
from fpavlm.nested_ridge import predict_ridge, select_ridge  # noqa: E402

BACKEND = "internvl2-2b"
PRIOR = ROOT / "experiments" / "results" / "nih_predictor_capacity_internvl2-2b.json"


def _lists():
    if not PRIOR.is_file():
        raise SystemExit("section 2.8 patient lists are not on disk")
    curve = json.loads(PRIOR.read_text(encoding="utf-8"))["curve"]
    lists = [row["patient_ids"] for row in curve]
    if [len(row) for row in lists] != list(PATIENT_COUNTS):
        raise ValueError("section 2.8 patient lists do not match the locked sizes")
    return lists


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=[BACKEND], default=BACKEND)
    args = parser.parse_args()
    if args.backend != BACKEND:
        raise SystemExit("section 2.10 fits internvl2-2b only")
    lists = _lists()
    torch.set_num_threads(4)
    torch.manual_seed(812)
    started = time.perf_counter()
    development = _records(DEVELOPMENT, REPLICATION_POSITIVE + REPLICATION_NEGATIVE)
    confirmation = _records(CONFIRMATION, CONFIRMATION_POSITIVE + CONFIRMATION_NEGATIVE)
    dev_ids = [row["patient_id"] for row in development]
    conf_ids = [row["patient_id"] for row in confirmation]
    full = lists[-1]
    if set(dev_ids) & set(conf_ids) or set(full) != set(dev_ids):
        raise ValueError("section 2.8 lists are not the development patients")
    if any(lists[i] != full[:count] for i, count in enumerate(PATIENT_COUNTS)):
        raise ValueError("section 2.8 lists are not nested")
    llm = load_backend(BACKEND)
    native = torch.cat([
        _native(llm, DEVELOPMENT, development, BACKEND),
        _native(llm, CONFIRMATION, confirmation, BACKEND),
    ]).float()
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if native.ndim != 3 or native.shape[1] != 256:
        raise ValueError("native tokens are not a 256-token grid")
    features = native.mean(dim=1).numpy()
    train_n = len(development)
    flat = native.reshape(len(native), -1).double()
    centered = flat - flat[:train_n].mean(0)
    basis = torch.linalg.svd(centered[:train_n], full_matrices=False).Vh.T[:, :RELEASE_RANK]
    oracle = (centered @ basis).float()
    tokens, hidden = int(native.shape[-2]), int(native.shape[-1])
    if features.shape != (len(native), hidden):
        raise ValueError("pooled features do not match the token width")
    digest = hashlib.sha256("\n".join(dev_ids).encode("utf-8")).hexdigest()

    def spec_for(name):
        return PrefixSpec(
            BACKEND, (RELEASE_RANK, 1), tokens, hidden, digest,
            _fingerprint({"release": name, "rank": RELEASE_RANK, "section": "2.10"}),
        )

    curve = []
    dev_index = {patient: i for i, patient in enumerate(dev_ids)}
    test = list(range(train_n, len(native)))
    for count, chosen in zip(PATIENT_COUNTS, lists):
        subset = [dev_index[patient] for patient in chosen]
        print(json.dumps({"training_patients": count}), flush=True)
        token_mean = native[subset].mean(0, keepdim=True)
        coordinate_mean = oracle[subset].mean(0, keepdim=True)
        model = select_ridge(features[subset], oracle[subset].numpy(), chosen, ALPHAS)
        predicted = torch.as_tensor(predict_ridge(features, model), dtype=torch.float32)
        coordinate = summarize(
            "pooled_coordinates", predicted[test], oracle[test], coordinate_mean, conf_ids,
            bootstrap_samples=2000)
        prefix = summarize(
            "pooled_prefix", fit_prefix(predicted, native, subset, test, spec_for(f"pooled-{count}")),
            native[test], token_mean, conf_ids, bootstrap_samples=2000)
        oracle_prefix = summarize(
            "oracle_prefix", fit_prefix(oracle, native, subset, test, spec_for(f"oracle-{count}")),
            native[test], token_mean, conf_ids, bootstrap_samples=2000)
        curve.append({
            "training_patients": count,
            "patient_ids": chosen,
            "alpha": None if model["intercept_only"] else model["alpha"],
            "intercept_only": model["intercept_only"],
            "alpha_at_boundary": model["alpha_at_boundary"],
            "coordinate": coordinate,
            "prefix": prefix,
            "oracle_prefix": oracle_prefix,
        })
    final = curve[-1]
    coordinate_hit = bool(final["coordinate"]["beats_mean_significantly"])
    prefix_hit = bool(final["prefix"]["beats_mean_significantly"])
    if prefix_hit:
        branch = "pooled_tokens_beat_the_token_mean"
    elif final["intercept_only"] or not coordinate_hit:
        branch = "no_coordinate_signal"
    else:
        branch = "coordinates_fit_but_prefix_does_not"
    result = {
        "schema": "nih_self_prediction_v1",
        "protocol_section": "2.10",
        "backend": BACKEND,
        "features": "mean_of_256_native_tokens",
        "unpooled_tokens_used_as_predictor": False,
        "feature_dim": hidden,
        "release_rank": RELEASE_RANK,
        "patient_counts": list(PATIENT_COUNTS),
        "patient_lists": "section_2.8",
        "penalty": "patient_leave_one_out_on_standardized_features",
        "prefix_ridge": RIDGE,
        "development_patients": len(development),
        "confirmation_patients": len(confirmation),
        "fit_saw_confirmation": False,
        "privacy_noise": False,
        "discretized": False,
        "adapter_trained": False,
        "pca_target_redefined": False,
        "qwen_fitted": False,
        "medgemma_ran": False,
        "official_test_list_read": False,
        "curve": curve,
        "intercept_only_at_72": bool(final["intercept_only"]),
        "coordinate_beats_mean": coordinate_hit,
        "prefix_beats_mean": prefix_hit,
        "branch": branch,
        "limitations": [
            "The predictor is the token mean. The unpooled matrix would make PCA a linear identity.",
            "The basis is fit once on all 72 development patients.",
            "The gate is the 72-patient row. An MSE change near 1e-8 is float residue of the intercept.",
            "A trainable adapter is not started from this result.",
            "The PCA target is not redefined from this result.",
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    destination = ROOT / "experiments" / "results" / "nih_self_prediction_internvl2-2b.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_curve(
        curve, ROOT / "figures" / "nih_internvl_pooled_pca_learning_curve",
        coordinate_ylim=None, predictor_label="Pooled InternVL2 tokens")
    print(json.dumps({
        "branch": branch,
        "intercept_only_at_72": result["intercept_only_at_72"],
        "coordinate_beats_mean": coordinate_hit,
        "prefix_beats_mean": prefix_hit,
        "coordinate": final["coordinate"]["mse_improvement_over_mean"],
        "prefix": final["prefix"]["mse_improvement_over_mean"],
        "oracle_prefix": final["oracle_prefix"]["mse_improvement_over_mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
