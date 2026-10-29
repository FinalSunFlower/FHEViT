"""Validated calibration metadata for FHE-aware decoding."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FHENoiseCalibration:
    method: str
    sample_count: int
    embedding_error_std: float
    embedding_rmse: float

    @classmethod
    def load(cls, path: str) -> "FHENoiseCalibration":
        resolved = os.path.abspath(os.fspath(path))
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"FHE noise calibration not found: {resolved}")
        with open(resolved, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("FHE noise calibration must be a JSON object")
        required = {"method", "sample_count", "embedding_error_std", "embedding_rmse"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"FHE noise calibration is missing fields: {missing}")
        calibration = cls(
            method=str(value["method"]),
            sample_count=int(value["sample_count"]),
            embedding_error_std=float(value["embedding_error_std"]),
            embedding_rmse=float(value["embedding_rmse"]),
        )
        if calibration.method != "paired_fhe_plain_release":
            raise ValueError(
                "noise calibration must compare released FHE embeddings with the "
                "matching plaintext circuit"
            )
        if calibration.sample_count < 2:
            raise ValueError("noise calibration requires at least two paired samples")
        if calibration.embedding_error_std <= 0 or calibration.embedding_rmse <= 0:
            raise ValueError("calibrated error statistics must be positive")
        return calibration


__all__ = ["FHENoiseCalibration"]
