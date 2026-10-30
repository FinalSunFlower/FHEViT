"""Does a frozen pretrained encoder predict the oracle PCA-4 directions?

The moment and Fourier probes fit their own features on 99 images. This control
replaces those features with TorchXRayVision DenseNet-121, trained elsewhere and
held fixed. Only a ridge from its 1024 pooled features to the four oracle
coordinates is learned. The penalty is chosen by leave-one-patient-out inside the
training patients, on standardized features. The test patients do not enter the
basis, the penalty, or either ridge.

The same fit is repeated on the first 4, 8, 12, 16 and 20 patients of one seeded
permutation. Every fraction is reported. The permutation is not chosen from the
test scores. The prefix ridge stays the protocol ridge; the nested penalty is only
the high-dimensional map that previously had no selected regularization.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import PATIENT_ROOT  # noqa: E402
from experiments.diagnose_release_sufficiency import _fingerprint, summarize  # noqa: E402
from experiments.download_patient_cohort import validate_manifest  # noqa: E402
from experiments.evaluate_diagnostic_prefix import extract  # noqa: E402
from fpavlm import InternVL2Adapter, PrefixSpec, Qwen2VLAdapter  # noqa: E402
from fpavlm.checkpoint import load_tensor_checkpoint  # noqa: E402
from fpavlm.diagnostic_prefix import DiagnosticPrefixProjector  # noqa: E402
from fpavlm.nested_ridge import predict_ridge, select_ridge  # noqa: E402
from fpavlm.provenance import file_sha256  # noqa: E402

WEIGHTS = "densenet121-res224-all"
PATIENT_COUNTS = (4, 8, 12, 16, 20)
ALPHAS = np.concatenate((np.logspace(-3, 6, 19), [np.inf]))
RELEASE_RANK = 4
RIDGE = 1.0


def densenet_features(paths):
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights=WEIGHTS)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoded = []
    with torch.no_grad():
        for path in paths:
            image = torch.as_tensor(xrv.utils.load_image(str(path)), dtype=torch.float32, device=device)
            if image.ndim != 3:
                raise ValueError("TorchXRayVision image must be [1,H,W]")
            _, height, width = image.shape
            side = min(int(height), int(width))
            top, left = (int(height) - side) // 2, (int(width) - side) // 2
            image = image[:, top:top + side, left:left + side].unsqueeze(0)
            image = torch.nn.functional.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
            features = model.features2(image)
            encoded.append(features.detach().float().cpu())
    stacked = torch.cat(encoded)
    if not torch.isfinite(stacked).all():
        raise RuntimeError("nonfinite DenseNet features")
    return stacked.numpy(), file_sha256(model.weights_filename_local)


def fit_prefix(codes, native, train, test, spec):
    release = torch.as_tensor(codes, dtype=torch.float32).reshape(len(codes), RELEASE_RANK, 1)
    projector = DiagnosticPrefixProjector(spec).fit(release[train], native[train], sigma=0., ridge=RIDGE)
    return projector(release[test])


def spearman(x, y):
    left = np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)
    right = np.argsort(np.argsort(np.asarray(y, dtype=float))).astype(float)
    if left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def plot_curve(curve, destination, coordinate_ylim=(-10, 10), predictor_label="Frozen DenseNet"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 1.0,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    counts = [row["training_patients"] for row in curve]
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))

    def series(key):
        values = np.array([row[key]["mse_improvement_over_mean"] * 100 for row in curve])
        lower = np.array([row[key]["mse_improvement_ci"]["ci95_lower"] * 100 for row in curve])
        upper = np.array([row[key]["mse_improvement_ci"]["ci95_upper"] * 100 for row in curve])
        return values, np.vstack((np.maximum(values - lower, 0), np.maximum(upper - values, 0)))

    def quiet(values, errors):
        if max(np.max(np.abs(values)), np.max(errors)) < 0.05:
            return np.zeros_like(values), np.zeros_like(errors)
        return values, errors

    prefix, prefix_error = quiet(*series("prefix"))
    oracle, oracle_error = series("oracle_prefix")
    coordinates, coordinate_error = quiet(*series("coordinate"))
    axes[0].axhline(0, color="#CFCECE", linewidth=1.5, zorder=0)
    axes[0].errorbar(counts, oracle, yerr=oracle_error, color="#42949E", marker="o",
                     linewidth=2, capsize=3, label="Oracle coordinates")
    axes[0].errorbar(counts, prefix, yerr=prefix_error, color="#0F4D92", marker="o",
                     linewidth=2, capsize=3, label=predictor_label)
    axes[0].set_xlabel("Training patients")
    axes[0].set_ylabel("Held-out MSE change vs mean (%)")
    axes[0].set_title("Prefix reconstruction", fontsize=11.5)
    axes[0].legend(fontsize=8.5)
    axes[1].axhline(0, color="#CFCECE", linewidth=1.5, zorder=0)
    axes[1].errorbar(counts, coordinates, yerr=coordinate_error, color="#0F4D92", marker="o",
                     linewidth=2, capsize=3, label=predictor_label)
    axes[1].set_xlabel("Training patients")
    axes[1].set_ylabel("Held-out coordinate fit (%)")
    axes[1].set_title("Oracle PCA-4 coordinates", fontsize=11.5)
    if coordinate_ylim is not None:
        axes[1].set_ylim(*coordinate_ylim)
    axes[1].legend(fontsize=8.5)
    for axis in axes:
        axis.set_xticks(counts)
        axis.tick_params(axis="both", labelsize=9)
    figure.text(
        0.5, 0.015,
        "Development-only diagnostic; not clinical performance or encrypted inference.",
        ha="center", va="bottom", fontsize=8,
    )
    figure.tight_layout(pad=0.6, rect=(0, 0.07, 1, 1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".png"), dpi=300)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["qwen2vl-2b", "internvl2-2b"], default="qwen2vl-2b")
    parser.add_argument("--remote-code-sha256")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("bootstrap-samples must be at least 100")
    torch.set_num_threads(4)
    started = time.perf_counter()
    evidence = json.loads((ROOT / "experiments/results/spatial_quadratic_diagnostic.json").read_text("utf-8"))
    digest = evidence["checkpoint_sha256"]
    checkpoint = load_tensor_checkpoint(str(ROOT / "artifacts/spatial_quadratic_diagnostic.pt"),
                                        expected_sha256=digest)
    records = json.loads((PATIENT_ROOT / "metadata.json").read_text("utf-8"))
    validate_manifest(records, PATIENT_ROOT)
    rows = records["records"]
    training, evaluation = set(checkpoint["training_patients"]), set(checkpoint["evaluation_patients"])
    if training & evaluation or len(training) != PATIENT_COUNTS[-1]:
        raise ValueError("the learning curve is declared for the 20 training patients")
    train = [i for i, row in enumerate(rows) if row["patient_id"] in training]
    test = [i for i, row in enumerate(rows) if row["patient_id"] in evaluation]
    if not train or not test or len(train) + len(test) != len(rows):
        raise ValueError("cohort is not exactly bound to the encoder patient split")
    patients_test = [rows[i]["patient_id"] for i in test]
    paths = [PATIENT_ROOT / row["relative_path"] for row in rows]
    order = np.random.default_rng(812).permutation(sorted(training)).tolist()
    features, weight_sha = densenet_features(paths)
    model_root = ROOT / "models" / args.backend
    llm = (Qwen2VLAdapter(str(model_root)) if args.backend == "qwen2vl-2b"
           else InternVL2Adapter(str(model_root), allow_remote_code=True,
                                 remote_code_sha256=args.remote_code_sha256))
    llm.load()
    native = torch.cat([extract(llm, path, args.backend) for path in paths]).float()
    tokens, hidden = native.shape[-2], native.shape[-1]
    flat = native.reshape(len(native), -1).double()
    centered = flat - flat[train].mean(0)
    basis = torch.linalg.svd(centered[train], full_matrices=False).Vh.T[:, :RELEASE_RANK]
    oracle = (centered @ basis).float()

    def spec_for(name):
        return PrefixSpec(args.backend, (RELEASE_RANK, 1), tokens, hidden, digest,
                          _fingerprint({"release": name, "rank": RELEASE_RANK}))

    curve = []
    for count in PATIENT_COUNTS:
        chosen = set(order[:count])
        subset = [i for i in train if rows[i]["patient_id"] in chosen]
        subset_patients = [rows[i]["patient_id"] for i in subset]
        token_mean = native[subset].mean(0, keepdim=True)
        coordinate_mean = oracle[subset].mean(0, keepdim=True)
        model = select_ridge(features[subset], oracle[subset].numpy(), subset_patients, ALPHAS)
        predicted = torch.as_tensor(predict_ridge(features, model), dtype=torch.float32)
        coordinate = summarize(
            "densenet_coordinates", predicted[test], oracle[test], coordinate_mean, patients_test,
            bootstrap_samples=args.bootstrap_samples)
        prefix = summarize(
            "densenet_prefix", fit_prefix(predicted, native, subset, test, spec_for(f"densenet-{count}")),
            native[test], token_mean, patients_test, bootstrap_samples=args.bootstrap_samples)
        oracle_prefix = summarize(
            "oracle_prefix", fit_prefix(oracle, native, subset, test, spec_for(f"oracle-{count}")),
            native[test], token_mean, patients_test, bootstrap_samples=args.bootstrap_samples)
        curve.append({
            "training_patients": count, "patient_ids": order[:count],
            "alpha": None if model["intercept_only"] else model["alpha"],
            "intercept_only": model["intercept_only"],
            "alpha_at_boundary": model["alpha_at_boundary"], "train_images": len(subset),
            "coordinate": coordinate, "prefix": prefix, "oracle_prefix": oracle_prefix,
        })
    gains = [row["prefix"]["mse_improvement_over_mean"] for row in curve]
    final = curve[-1]["prefix"]
    intercepts = all(row["intercept_only"] for row in curve)
    rising = bool(not intercepts and gains[-1] > gains[0])
    monotone = bool(not intercepts and all(later >= earlier for earlier, later in zip(gains, gains[1:])))
    if intercepts:
        verdict = ("At every training size, patient-level cross-validation selects the intercept. "
                   "Frozen DenseNet features do not predict this PCA-4 target beyond the training mean.")
    elif final["beats_mean_significantly"] and rising:
        verdict = ("Held-out reconstruction beats the constant mean, and the 20-patient fit "
                   "is higher than the 4-patient fit.")
    elif rising:
        verdict = ("Held-out reconstruction improves as training patients are added, and at "
                   "20 patients it still does not beat the constant mean.")
    else:
        verdict = ("Held-out reconstruction does not improve as training patients are added. "
                   "Frozen DenseNet features do not show a learnable map onto this PCA-4 target.")
    result = {
        "schema": "pretrained_anchor_learning_curve_v1", "evidence_scope": "development_only",
        "backend": args.backend, "encoder": WEIGHTS, "encoder_sha256": weight_sha,
        "feature_dim": int(features.shape[1]), "encoder_trained": False,
        "penalty": "patient_leave_one_out_on_standardized_features",
        "alphas": [None if np.isinf(value) else float(value) for value in ALPHAS],
        "patient_order_seed": 812, "patient_order": order,
        "patient_counts": list(PATIENT_COUNTS), "release_rank": RELEASE_RANK,
        "prefix_ridge": RIDGE, "curve": curve,
        "spearman_prefix_vs_patients": None if intercepts else spearman(PATIENT_COUNTS, gains),
        "rising": rising, "monotone": monotone,
        "final_beats_mean": bool(final["beats_mean_significantly"]),
        "gate_passed": bool(final["beats_mean_significantly"]), "verdict": verdict,
        "fhe_executed": False, "disease_head_used": False,
        "bootstrap_samples": args.bootstrap_samples,
        "limitations": [
            "Development cohort only; the curve reuses the same seven test patients at every point.",
            "The encoder is frozen and plaintext. This is not an encrypted release.",
            "The PCA basis is fit once on all 20 training patients so the target stays fixed.",
            "No fraction was selected after seeing the test scores.",
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    destination = ROOT / "experiments/results" / f"pretrained_anchor_learning_curve_{args.backend}.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figure = ROOT / "figures" / "densenet_pca_learning_curve"
    plot_curve(curve, figure)
    print(json.dumps({
        "verdict": verdict, "rising": rising, "monotone": monotone,
        "curve": [{"patients": row["training_patients"], "alpha": row["alpha"],
                   "intercept_only": row["intercept_only"],
                   "prefix": row["prefix"]["mse_improvement_over_mean"],
                   "prefix_ci": [row["prefix"]["mse_improvement_ci"]["ci95_lower"],
                                 row["prefix"]["mse_improvement_ci"]["ci95_upper"]],
                   "coordinate": row["coordinate"]["mse_improvement_over_mean"],
                   "oracle_prefix": row["oracle_prefix"]["mse_improvement_over_mean"]}
                  for row in curve],
    }, indent=2))


if __name__ == "__main__":
    main()
