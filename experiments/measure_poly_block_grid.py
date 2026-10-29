"""Time one polynomial block across the locked width and token grid.

EVALUATION_PROTOCOL.md section 2.34. A failed pair is not replaced, and the
slopes are not a full InternViT time. The depth budget stays 6.
"""
from __future__ import annotations

import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpavlm.cipher_native.canr import ClientCANREngine, ServerCANREngine  # noqa: E402
from fpavlm.cipher_native.ciphertext_tensor import CiphertextTensor  # noqa: E402
from fpavlm.cipher_native.med_fhe_vit_block import MedFHEBlock  # noqa: E402

DESTINATION = ROOT / "experiments" / "results" / "poly_block_ckks_grid.json"
WIDTHS = (32, 64, 128)
TOKENS = (4, 8, 16)
DEPTH_BUDGET = 6


def log_slope(sizes, times):
    """Ordinary least-squares slope of log time on log size."""
    if len(sizes) != len(times) or len(sizes) < 2:
        raise ValueError("a slope needs two paired points")
    if any(size <= 0 or observed <= 0 for size, observed in zip(sizes, times)):
        raise ValueError("log slope needs positive sizes and times")
    x = [math.log(float(size)) for size in sizes]
    y = [math.log(float(observed)) for observed in times]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    xx = sum((item - x_mean) ** 2 for item in x)
    xy = sum((left - x_mean) * (right - y_mean) for left, right in zip(x, y))
    return xy / xx


def _block(width, tokens):
    return MedFHEBlock(
        dim=width,
        num_tokens=tokens,
        activation_type="polyact",
        poly_act_degree=1,
        poly_norm_degree=2,
        enable_token_mix=True,
        use_cadn=False,
        num_regions=1,
    )


def _write(payload):
    DESTINATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _slope(points, *, fixed_tokens=None, fixed_width=None):
    chosen = [
        point for point in points
        if point["status"] == "ok"
        and (fixed_tokens is None or point["tokens"] == fixed_tokens)
        and (fixed_width is None or point["width"] == fixed_width)
    ]
    expected = len(WIDTHS) if fixed_tokens is not None else len(TOKENS)
    if len(chosen) != expected:
        return None
    if fixed_tokens is not None:
        chosen.sort(key=lambda point: point["width"])
        return log_slope([point["width"] for point in chosen], [point["forward_s"] for point in chosen])
    chosen.sort(key=lambda point: point["tokens"])
    return log_slope([point["tokens"] for point in chosen], [point["forward_s"] for point in chosen])


def main():
    torch.manual_seed(812)
    payload = {
        "schema": "poly_block_ckks_grid_v1",
        "protocol_section": "2.34",
        "widths": list(WIDTHS),
        "tokens": list(TOKENS),
        "poly_modulus_degree": 32768,
        "scale_bits": 28,
        "depth_budget": DEPTH_BUDGET,
        "full_internvit_time_reported": False,
        "confirmation_opened": False,
        "python": platform.python_version(),
        "points": [],
    }
    try:
        import tenseal as ts

        payload["tenseal"] = ts.__version__
        started = time.perf_counter()
        client = ClientCANREngine(
            poly_modulus_degree=32768,
            scale_bits=28,
            depth_budget=DEPTH_BUDGET,
            generate_relin_keys=True,
        )
        payload["context_s"] = time.perf_counter() - started
        server = ServerCANREngine(client)
    except Exception as exc:  # noqa: BLE001 — context failure is the recorded anchor
        payload["status"] = "context_failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        _write(payload)
        raise SystemExit(payload["status"]) from exc
    for tokens in TOKENS:
        for width in WIDTHS:
            point = {"width": width, "tokens": tokens, "status": "failed"}
            try:
                block = _block(width, tokens)
                point["multiplicative_depth"] = int(block.multiplicative_depth())
                if point["multiplicative_depth"] != DEPTH_BUDGET:
                    point["status"] = "depth_mismatch"
                    payload["points"].append(point)
                    _write(payload)
                    continue
                block.prepare_cipher_weights()
                torch.manual_seed(812)
                values = torch.randn(tokens, width)
                encrypted = CiphertextTensor(
                    ciphertexts=client.encrypt(values), engine=server, depth=0,
                )
                forward_started = time.perf_counter()
                output = block(encrypted)
                point["forward_s"] = time.perf_counter() - forward_started
                point["output_depth"] = int(output.depth)
                decrypted = client.decrypt(output._cipher, dim=width)
                point["output_finite"] = bool(torch.isfinite(decrypted).all())
                point["status"] = "ok" if point["output_finite"] else "nonfinite"
            except Exception as exc:  # noqa: BLE001 — a failed pair is not replaced
                point["status"] = "failed"
                point["error"] = f"{type(exc).__name__}: {exc}"
                point["traceback"] = traceback.format_exc()
            payload["points"].append(point)
            _write(payload)
            print(json.dumps({
                "width": width,
                "tokens": tokens,
                "status": point["status"],
                "forward_s": point.get("forward_s"),
                "output_depth": point.get("output_depth"),
            }), flush=True)
    payload["width_slope_at_4_tokens"] = _slope(payload["points"], fixed_tokens=4)
    payload["token_slope_at_width_64"] = _slope(payload["points"], fixed_width=64)
    payload["status"] = "ok"
    payload["full_internvit_time_reported"] = False
    _write(payload)
    print(json.dumps({
        "width_slope_at_4_tokens": payload["width_slope_at_4_tokens"],
        "token_slope_at_width_64": payload["token_slope_at_width_64"],
    }), flush=True)


if __name__ == "__main__":
    main()
