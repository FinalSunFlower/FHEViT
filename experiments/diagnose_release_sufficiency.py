"""Phase-0 plaintext sufficiency diagnostic: where does the image signal die?

The 0.3 development record shows that the DP release prefix barely beats a
constant training-mean prefix in feature MSE (Qwen 0.042%, InternVL 0.159%),
which raises the question the whole pipeline hinges on: does the release ->
bridge step carry ANY statistically significant image-dependent signal, or is
the causal chain broken before any language backend is involved?

This script isolates that question with NO FHE and NO privacy noise required.
Using the SAME linear ridge bridge and the SAME patient-disjoint split as the
0.3 evaluation, it fits the bridge from several release representations to the
frozen native visual tokens and measures, with patient-level bootstrap intervals,
whether each release beats the constant-mean prefix:

  * ``constant_mean``      the image-independent baseline (improvement is 0 by construction);
  * ``pca_rank_{r}``       an ORACLE upper bound: the native tokens projected onto their own
                           top-r training PCA directions. This is NOT an achievable encrypted
                           release; it answers "can the linear bridge use r image-dependent
                           dimensions at all?" and locates the information cliff by rank;
  * ``quadratic_clean``    the deployed 0.3 low-rank quadratic release, no noise;
  * ``quadratic_dp``       the deployed 0.3 release plus the accounted Gaussian DP noise.

Reading: if even a high-rank ORACLE fails to beat the mean, the bottleneck is the
linear bridge / small cohort (possibility B); if the oracle beats the mean but the
quadratic_clean release does not, the low-rank release itself discards the signal
(possibility A). This diagnostic does not measure clinical accuracy or the 0.4
finite-state codec, which needs real regional concept supervision to train.
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

from experiments.common import PATIENT_ROOT, preprocess_image  # noqa: E402
from experiments.download_patient_cohort import validate_manifest  # noqa: E402
from experiments.evaluate_diagnostic_prefix import extract  # noqa: E402
from fpavlm import (  # noqa: E402
    DiagnosticPrefixProjector,
    InternVL2Adapter,
    PrefixSpec,
    Qwen2VLAdapter,
    SpatialQuadraticRelease,
)
from fpavlm.checkpoint import load_tensor_checkpoint  # noqa: E402
from fpavlm.cipher_native.dp_accountant import l2_clip, sigma_for_target_epsilon  # noqa: E402
from fpavlm.evaluation import paired_metric_advantage  # noqa: E402
from fpavlm.provenance import file_sha256  # noqa: E402


def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def per_study_metrics(prediction, reference, train_mean):
    """Per-study cosine and MSE-improvement-over-mean; higher is better."""
    pred = prediction.flatten(1).double()
    ref = reference.flatten(1).double()
    mean = train_mean.flatten(1).double()
    residual = (pred - ref).square().sum(1)
    total = (ref - mean).square().sum(1).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(pred, ref, dim=1)
    mean_cosine = torch.nn.functional.cosine_similarity(mean.expand_as(ref), ref, dim=1)
    return {"cosine": cosine, "mean_cosine": mean_cosine, "mse_improvement": 1 - residual / total}


def summarize(name, prediction, reference, train_mean, patients, *, bootstrap_samples):
    metrics = per_study_metrics(prediction, reference, train_mean)
    mse = metrics["mse_improvement"].tolist()
    cosine = metrics["cosine"].tolist()
    # Advantage over the constant-mean prefix. mse_improvement is already relative to the
    # mean, so its own baseline is a vector of zeros; cosine is compared to the mean cosine.
    mse_ci = paired_metric_advantage(mse, [0.0] * len(mse), patients, bootstrap_samples=bootstrap_samples)
    cosine_ci = paired_metric_advantage(cosine, metrics["mean_cosine"].tolist(), patients,
                                        bootstrap_samples=bootstrap_samples)
    return {"release": name,
            "cosine_mean": float(metrics["cosine"].mean()),
            "mse_improvement_over_mean": float(metrics["mse_improvement"].mean()),
            "mse_improvement_ci": mse_ci,
            "cosine_advantage_ci": cosine_ci,
            "beats_mean_significantly": bool(mse_ci["significant_positive"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["qwen2vl-2b", "internvl2-2b"], default="qwen2vl-2b")
    parser.add_argument("--remote-code-sha256", help="reviewed local InternVL source digest")
    parser.add_argument("--epsilon", type=float, default=3.)
    parser.add_argument("--pca-ranks", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.epsilon <= 0 or any(r < 1 for r in args.pca_ranks) or args.bootstrap_samples < 100:
        parser.error("epsilon and PCA ranks must be positive and bootstrap-samples >= 100")
    torch.set_num_threads(4)
    torch.manual_seed(812)
    started = time.perf_counter()

    evidence = json.loads((ROOT / "experiments/results/spatial_quadratic_diagnostic.json").read_text("utf-8"))
    digest = evidence["checkpoint_sha256"]
    checkpoint = load_tensor_checkpoint(str(ROOT / "artifacts/spatial_quadratic_diagnostic.pt"),
                                        expected_sha256=digest)
    manifest_path = PATIENT_ROOT / "metadata.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    validate_manifest(manifest, PATIENT_ROOT)
    records = manifest["records"]
    training = set(checkpoint["training_patients"])
    evaluation = set(checkpoint["evaluation_patients"])
    if training & evaluation:
        raise ValueError("training and evaluation patients overlap")
    train = [i for i, r in enumerate(records) if r["patient_id"] in training]
    test = [i for i, r in enumerate(records) if r["patient_id"] in evaluation]
    if not train or not test or len(train) + len(test) != len(records):
        raise ValueError("cohort is not exactly bound to the encoder patient split")
    patients_test = [records[i]["patient_id"] for i in test]

    model = SpatialQuadraticRelease(checkpoint["patch_dim"], checkpoint["latent_dim"], checkpoint["rank"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    paths = [PATIENT_ROOT / r["relative_path"] for r in records]
    with torch.no_grad():
        images = torch.cat([preprocess_image(p) for p in paths])
        patches = images.permute(0, 2, 3, 1).reshape(-1, 4, 56, 4, 56, 3)
        patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(-1, 16, 9408)
        clean = l2_clip(model(patches), clip_norm=1.)
    release_shape = tuple(clean.shape[1:])
    sigma, _, _ = sigma_for_target_epsilon(args.epsilon, delta=1e-5, dim=int(clean[0].numel()), clip_norm=1.)

    model_root = ROOT / "models" / args.backend
    if args.backend == "qwen2vl-2b":
        llm = Qwen2VLAdapter(str(model_root))
    else:
        llm = InternVL2Adapter(str(model_root), allow_remote_code=True,
                               remote_code_sha256=args.remote_code_sha256)
    llm.load()
    native = torch.cat([extract(llm, path, args.backend) for path in paths]).float()
    hidden = native.shape[-1]
    tokens = native.shape[-2]
    train_mean = native[train].mean(0, keepdim=True)

    def bridge_predict(release, spec, *, sigma_fit, sigma_eval):
        fitted = DiagnosticPrefixProjector(spec).fit(release[train], native[train], sigma=sigma_fit)
        evaluated = release[test]
        if sigma_eval:
            generator = torch.Generator().manual_seed(1941)
            evaluated = evaluated + sigma_eval * torch.randn(evaluated.shape, generator=generator)
        return fitted(evaluated)

    summaries = []
    # 1) constant-mean prefix (image-independent baseline).
    summaries.append(summarize("constant_mean", train_mean.expand(len(test), -1, -1),
                               native[test], train_mean, patients_test,
                               bootstrap_samples=args.bootstrap_samples))

    # 2) ORACLE PCA-rank upper bounds: native projected onto its own training PCA basis.
    native_flat = native.reshape(len(native), -1).double()
    centered = native_flat - native_flat[train].mean(0, keepdim=True)
    basis = torch.linalg.svd(centered[train], full_matrices=False).Vh.T
    for rank in sorted({min(r, basis.shape[1]) for r in args.pca_ranks}):
        code = (centered @ basis[:, :rank]).reshape(len(native), rank, 1).float()
        spec = PrefixSpec(args.backend, (rank, 1), tokens, hidden, digest,
                          _fingerprint({"release": "pca_oracle", "rank": rank}))
        prediction = bridge_predict(code, spec, sigma_fit=0., sigma_eval=0.)
        summaries.append(summarize(f"pca_rank_{rank}", prediction, native[test], train_mean,
                                   patients_test, bootstrap_samples=args.bootstrap_samples))

    # 3) deployed 0.3 low-rank quadratic release, clean and with accounted DP noise.
    quad_spec = PrefixSpec(args.backend, release_shape, tokens, hidden, digest,
                           _fingerprint({"release": "spatial_quadratic", "clip_norm": 1.}))
    summaries.append(summarize("quadratic_clean",
                               bridge_predict(clean, quad_spec, sigma_fit=0., sigma_eval=0.),
                               native[test], train_mean, patients_test,
                               bootstrap_samples=args.bootstrap_samples))
    dp_spec = PrefixSpec(args.backend, release_shape, tokens, hidden, digest,
                         _fingerprint({"release": "spatial_quadratic_dp", "epsilon": args.epsilon,
                                       "delta": 1e-5, "sigma_per_coord": sigma}))
    summaries.append(summarize("quadratic_dp",
                               bridge_predict(clean, dp_spec, sigma_fit=sigma, sigma_eval=sigma),
                               native[test], train_mean, patients_test,
                               bootstrap_samples=args.bootstrap_samples))

    oracle_beats = [s for s in summaries if s["release"].startswith("pca_rank_") and s["beats_mean_significantly"]]
    quad_clean = next(s for s in summaries if s["release"] == "quadratic_clean")
    if oracle_beats and not quad_clean["beats_mean_significantly"]:
        localization = ("release_fidelity (A): a linear bridge from an image-dependent code CAN beat "
                        "the mean, but the deployed low-rank quadratic release cannot")
    elif not oracle_beats:
        localization = ("bridge_or_cohort (B): not even an oracle low-rank image code beats the mean "
                        "through this linear bridge on this cohort")
    else:
        localization = ("release_carries_signal: the deployed clean release already beats the mean; "
                        "inspect the DP-noise step")

    result = {"schema": "release_sufficiency_diagnostic_v1", "evidence_scope": "development_only",
              "backend": args.backend, "encoder_sha256": digest,
              "cohort_manifest_sha256": file_sha256(manifest_path),
              "training_patients": sorted(training), "evaluation_patients": sorted(evaluation),
              "train_samples": len(train), "test_samples": len(test), "patient_disjoint": True,
              "release_shape": list(release_shape), "native_tokens": tokens, "native_hidden": hidden,
              "epsilon": args.epsilon, "delta": 1e-5, "sigma_per_coord": sigma,
              "noise_rms_l2": sigma * (int(clean[0].numel()) ** .5),
              "bootstrap_samples": args.bootstrap_samples, "releases": summaries,
              "localization": localization,
              "limitations": ["Small development cohort; not sealed or externally confirmed.",
                              "PCA-rank rows are an oracle upper bound, not an achievable encrypted release.",
                              "Linear ridge bridge only; a higher-capacity bridge is untested here.",
                              "This measures feature transfer, not clinical accuracy or report factuality.",
                              "The 0.4 finite-state codec is not evaluated; it needs real regional concepts."],
              "elapsed_s": time.perf_counter() - started}
    destination = ROOT / "experiments/results" / f"release_sufficiency_{args.backend}.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"backend": args.backend, "localization": localization,
                      "releases": [{"release": s["release"],
                                    "mse_improvement_over_mean": s["mse_improvement_over_mean"],
                                    "ci95": [s["mse_improvement_ci"]["ci95_lower"],
                                             s["mse_improvement_ci"]["ci95_upper"]],
                                    "beats_mean": s["beats_mean_significantly"]} for s in summaries]}, indent=2))


if __name__ == "__main__":
    main()
