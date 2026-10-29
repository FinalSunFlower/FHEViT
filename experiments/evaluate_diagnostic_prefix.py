"""Patient-disjoint development fit of the quadratic release to local VLMs.

Native towers are offline teachers. Runtime bridges consume only the client's
sanitized release. Feature similarity and report examples are not clinical
non-inferiority; a training-mean control exposes image-independent predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import PATIENT_ROOT, preprocess_image  # noqa: E402
from experiments.download_patient_cohort import validate_manifest  # noqa: E402
from fpavlm import (  # noqa: E402
    DiagnosticPrefixProjector,
    InternVL2Adapter,
    PrefixSpec,
    Qwen2VLAdapter,
    SpatialQuadraticRelease,
)
from fpavlm.checkpoint import load_tensor_checkpoint  # noqa: E402
from fpavlm.cipher_native.dp_accountant import l2_clip, sigma_for_target_epsilon  # noqa: E402
from fpavlm.provenance import directory_fingerprint, file_sha256  # noqa: E402

PROMPT = "Describe the findings visible in this chest X-ray in a brief radiology report."


def qwen_inputs(llm, path, prompt):
    # Fix the spatial layout before the tower; never interpolate flattened tokens.
    with Image.open(path) as source:
        rgb = source.convert("RGB").resize((448, 448), Image.Resampling.BILINEAR)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": rgb}, {"type": "text", "text": prompt},
    ]}]
    text = llm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return llm.processor(text=[text], images=[rgb], return_tensors="pt").to("cuda", dtype=torch.bfloat16)


@torch.no_grad()
def extract(llm, path, backend):
    if backend == "qwen2vl-2b":
        inputs = qwen_inputs(llm, path, ".")
        if inputs["image_grid_thw"].tolist() != [[1, 32, 32]]:
            raise ValueError("unexpected Qwen spatial grid")
        tokens = llm.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]).unsqueeze(0)
    else:
        tokens = llm.extract_native_visual_features(str(path))
    if tokens is None or tuple(tokens.shape) != (1, 256, llm.llm_hidden_dim):
        raise ValueError("native visual tokens do not match the fixed 16x16 backend contract")
    return tokens.float().cpu()


def feature_metrics(prediction, reference, training_mean):
    residual = (prediction - reference).flatten(1)
    total = (reference - training_mean).square().sum().clamp_min(1e-12)
    return {"cosine_mean": float(torch.nn.functional.cosine_similarity(
                prediction.flatten(1), reference.flatten(1)).mean()),
            "relative_l2_mean": float((residual.norm(dim=1) / reference.flatten(1).norm(dim=1)).mean()),
            "mse_improvement_over_training_mean": float(1 - residual.square().sum() / total)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["qwen2vl-2b", "internvl2-2b"], default="qwen2vl-2b")
    parser.add_argument("--remote-code-sha256", help="reviewed local InternVL source digest")
    parser.add_argument("--epsilon", type=float, default=3.)
    parser.add_argument("--report-patients", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    if args.report_patients < 0 or args.max_new_tokens < 1:
        parser.error("report-patients must be nonnegative and max-new-tokens positive")
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
    model = SpatialQuadraticRelease(checkpoint["patch_dim"], checkpoint["latent_dim"], checkpoint["rank"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    paths = [PATIENT_ROOT / r["relative_path"] for r in records]
    with torch.no_grad():
        images = torch.cat([preprocess_image(p) for p in paths])
        patches = images.permute(0, 2, 3, 1).reshape(-1, 4, 56, 4, 56, 3)
        patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(-1, 16, 9408)
        clean = l2_clip(model(patches), clip_norm=1.)
    sigma, _, _ = sigma_for_target_epsilon(args.epsilon, delta=1e-5, dim=20, clip_norm=1.)
    mechanism = {"kind": "whole_image_isotropic_gaussian", "clip_norm": 1., "epsilon": args.epsilon,
                 "delta": 1e-5, "sigma_per_coord": sigma, "dim": 20, "quantize": False}
    fingerprint = hashlib.sha256(json.dumps(mechanism, sort_keys=True, allow_nan=False).encode()).hexdigest()
    model_root = ROOT / "models" / args.backend
    if args.backend == "qwen2vl-2b":
        llm = Qwen2VLAdapter(str(model_root))
    else:
        llm = InternVL2Adapter(str(model_root), allow_remote_code=True,
                              remote_code_sha256=args.remote_code_sha256)
    llm.load()
    native = []
    for i, path in enumerate(paths):
        native.append(extract(llm, path, args.backend))
        if (i + 1) % 20 == 0:
            print(json.dumps({"extracted": i + 1, "total": len(paths)}), flush=True)
    native = torch.cat(native)
    spec = PrefixSpec(args.backend, (5, 4), 256, llm.llm_hidden_dim, digest, fingerprint)
    projectors = {}
    artifacts = {}
    for name, noise in (("clean", 0.), ("dp", sigma)):
        fitted = DiagnosticPrefixProjector(spec).fit(clean[train], native[train], sigma=noise)
        projectors[name] = fitted
        artifact = ROOT / "artifacts/adapters" / f"{args.backend}-diagnostic-{name}.pt"
        fitted.save(str(artifact), training_patient_ids=sorted(training))
        artifacts[name] = {"path": artifact.relative_to(ROOT).as_posix(), "sha256": file_sha256(artifact),
                           "training_sigma": noise}
    # Seeded noise is for reproducible offline measurement only, never deployment.
    generator = torch.Generator().manual_seed(1941)
    noisy = clean[test] + sigma * torch.randn(clean[test].shape, generator=generator)
    mean = native[train].mean(0, keepdim=True)
    predictions = {"clean": projectors["clean"](clean[test]), "dp": projectors["dp"](noisy),
                   "training_mean": mean.expand(len(test), -1, -1), "native_injected": native[test]}
    metrics = {k: feature_metrics(v, native[test], mean) for k, v in predictions.items()}
    result = {"status": "running", "evidence_scope": "development_only", "backend": args.backend,
              "encoder_sha256": digest, "cohort_manifest_sha256": file_sha256(manifest_path),
              "model_config_sha256": file_sha256(model_root / "config.json"),
              "model_snapshot_sha256": directory_fingerprint(model_root),
              "remote_code_sha256": args.remote_code_sha256,
              "training_patients": sorted(training), "evaluation_patients": sorted(evaluation),
              "train_samples": len(train), "test_samples": len(test), "patient_disjoint": True,
              "mechanism": mechanism, "mechanism_fingerprint": fingerprint, "artifacts": artifacts,
              "feature_metrics": metrics, "reports": [], "clinical_noninferiority_evaluated": False,
              "limitations": ["Small development cohort; not sealed or externally confirmed.",
                              "Native and bridge towers use different input resolutions (448 and 224).",
                              "Fixed square image resizing; not full dynamic-resolution native VLM.",
                              "Injected native features are not the official native generation path.",
                              "Gaussian seed is public for experiment reproducibility only.",
                              "No CKKS is simulated here; its measured error is reported separately.",
                              "Report examples and feature similarity are not clinical accuracy."]}
    destination = ROOT / "experiments/results" / f"diagnostic_prefix_{args.backend}.json"

    def save():
        result["elapsed_s"] = time.perf_counter() - started
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    seen = set()
    for position, index in enumerate(test):
        patient = records[index]["patient_id"]
        if patient in seen:
            continue
        if len(seen) >= args.report_patients:
            break
        seen.add(patient)
        row = {"patient_id": patient, "relative_path": records[index]["relative_path"], "outputs": {}}
        for mode, features in predictions.items():
            start = time.perf_counter()
            text = llm.generate(features[position:position + 1], prompt=PROMPT,
                                max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1, seed=42)
            row["outputs"][mode] = {"text": text, "generation_s": time.perf_counter() - start}
        if args.backend == "qwen2vl-2b":
            inputs = qwen_inputs(llm, paths[index], PROMPT)
            start = time.perf_counter()
            with torch.no_grad():
                tokens = llm.model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = llm.tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            row["outputs"]["native_processor_448"] = {"text": text, "generation_s": time.perf_counter() - start}
        result["reports"].append(row)
        save()
        print(json.dumps({"reports_completed": len(seen), "backend": args.backend}), flush=True)
    result["status"] = "completed"
    result["prompt"] = PROMPT
    result["max_new_tokens"] = args.max_new_tokens
    save()
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
