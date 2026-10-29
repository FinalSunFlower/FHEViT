"""Time one reduced-token polynomial block under TenSEAL.

EVALUATION_PROTOCOL.md section 2.33. The depth budget stays 6. A failure
is recorded and the budget is not raised.
"""
from __future__ import annotations

import json
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

DESTINATION = ROOT / "experiments" / "results" / "poly_block_ckks_anchor.json"
TOKENS = 4
DIM = 64
DEPTH_BUDGET = 6


def _block():
    return MedFHEBlock(
        dim=DIM,
        num_tokens=TOKENS,
        activation_type="polyact",
        poly_act_degree=1,
        poly_norm_degree=2,
        enable_token_mix=True,
        use_cadn=False,
        num_regions=1,
    )


def main():
    torch.manual_seed(812)
    block = _block()
    depth = int(block.multiplicative_depth())
    payload = {
        "schema": "poly_block_ckks_anchor_v1",
        "protocol_section": "2.33",
        "block": "MedFHEBlock",
        "dim": DIM,
        "tokens": TOKENS,
        "poly_act_degree": 1,
        "poly_norm_degree": 2,
        "token_mix": True,
        "cadn": False,
        "poly_modulus_degree": 32768,
        "scale_bits": 28,
        "depth_budget": DEPTH_BUDGET,
        "relin_keys": True,
        "seed": 812,
        "multiplicative_depth": depth,
        "confirmation_opened": False,
        "python": platform.python_version(),
    }
    if depth != DEPTH_BUDGET:
        payload["status"] = "depth_mismatch"
        payload["forward_s"] = None
        DESTINATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("the block depth is not the locked budget")
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
        block.prepare_cipher_weights()
        values = torch.randn(TOKENS, DIM)
        encrypted = CiphertextTensor(
            ciphertexts=client.encrypt(values), engine=server, depth=0,
        )
        forward_started = time.perf_counter()
        output = block(encrypted)
        payload["forward_s"] = time.perf_counter() - forward_started
        payload["output_depth"] = int(output.depth)
        decrypted = client.decrypt(output._cipher, dim=DIM)
        payload["output_finite"] = bool(torch.isfinite(decrypted).all())
        payload["status"] = "ok" if payload["output_finite"] else "nonfinite"
    except Exception as exc:  # noqa: BLE001 — a failed forward is the recorded anchor
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
    DESTINATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "forward_s": payload.get("forward_s"),
        "output_depth": payload.get("output_depth"),
        "context_s": payload.get("context_s"),
        "multiplicative_depth": depth,
    }), flush=True)
    if payload["status"] not in {"ok", "failed"}:
        raise SystemExit(payload["status"])


if __name__ == "__main__":
    main()
