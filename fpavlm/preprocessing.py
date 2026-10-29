"""Canonical image preprocessing shared by deployment and experiments."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class ImagePreprocessingSpec:
    version: str = "fhevit_rgb_bilinear_symmetric_v1"
    image_size: int = 224
    input_range: float = 3.0
    color_mode: str = "RGB"
    interpolation: str = "bilinear"

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preprocess_image(
    image_path: Union[str, Path],
    *,
    image_size: int = 224,
    input_range: float = 3.0,
) -> torch.Tensor:
    """Return ``[1, 3, H, W]`` float32 input in ``[-input_range, input_range]``."""
    if image_size < 1:
        raise ValueError("image_size must be positive")
    if not np.isfinite(input_range) or input_range <= 0:
        raise ValueError("input_range must be finite and positive")
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
    array = ((array - 0.5) / 0.5) * float(input_range)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)


__all__ = ["ImagePreprocessingSpec", "preprocess_image"]
