"""Count multiplicative depth before any student AUC.

EVALUATION_PROTOCOL.md section 2.35. Students above depth 5 are not scored.
This file does not load a model and does not open the confirmation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DESTINATION = ROOT / "experiments" / "results" / "nih_student_depth.json"
BLOCK_DEPTH = 5
MAX_DEPTH = 5


def student_plans():
    """Locked depths. Layer stacks use one measured block per layer."""
    plans = []
    for layers in (2, 4, 8):
        depth = BLOCK_DEPTH * layers
        plans.append({
            "name": f"internvit_layers_{layers}",
            "depth": depth,
            "circuit_candidate": False,
            "score_auc": False,
            "reason": "above the depth limit",
        })
    plans.extend([
        {
            "name": "pixel_linear",
            "depth": 1,
            "circuit_candidate": True,
            "score_auc": True,
            "reason": "one ciphertext-plaintext map",
        },
        {
            "name": "pixel_square",
            "depth": 2,
            "circuit_candidate": True,
            "score_auc": True,
            "reason": "elementwise square then one linear map",
        },
        {
            "name": "layer1_linear",
            "depth": 1,
            "circuit_candidate": False,
            "score_auc": True,
            "reason": "attribution control from the first block mean",
        },
        {
            "name": "layer1_square",
            "depth": 2,
            "circuit_candidate": False,
            "score_auc": True,
            "reason": "attribution control from the first block mean",
        },
    ])
    for plan in plans:
        if plan["depth"] > MAX_DEPTH:
            plan["circuit_candidate"] = False
            plan["score_auc"] = False
        if plan["circuit_candidate"] and plan["depth"] > MAX_DEPTH:
            raise ValueError("a circuit candidate exceeds the depth limit")
    return plans


def main():
    plans = student_plans()
    payload = {
        "schema": "nih_student_depth_v1",
        "protocol_section": "2.35",
        "parameter_set": {"poly_modulus_degree": 32768, "scale_bits": 28, "depth_budget": 6},
        "block_output_depth": BLOCK_DEPTH,
        "max_circuit_depth": MAX_DEPTH,
        "tenseal_bootstraps": False,
        "plans": plans,
        "scored_names": [plan["name"] for plan in plans if plan["score_auc"]],
        "circuit_names": [plan["name"] for plan in plans if plan["circuit_candidate"]],
        "confirmation_opened": False,
        "auc_computed": False,
    }
    DESTINATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "scored_names": payload["scored_names"],
        "circuit_names": payload["circuit_names"],
        "unscored": [plan["name"] for plan in plans if not plan["score_auc"]],
    }), flush=True)


if __name__ == "__main__":
    main()
