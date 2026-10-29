from __future__ import annotations

import numpy as np
from PIL import Image

from fpavlm.preprocessing import ImagePreprocessingSpec, preprocess_image


def test_canonical_preprocessing_range_and_shape(tmp_path):
    path = tmp_path / "constant.png"
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:, 4:, :] = 255
    Image.fromarray(pixels).save(path)

    tensor = preprocess_image(path, image_size=4, input_range=3.0)

    assert tensor.shape == (1, 3, 4, 4)
    assert float(tensor.min()) == -3.0
    assert float(tensor.max()) == 3.0


def test_preprocessing_fingerprint_binds_configuration():
    base = ImagePreprocessingSpec()
    assert base.fingerprint() == ImagePreprocessingSpec().fingerprint()
    assert base.fingerprint() != ImagePreprocessingSpec(input_range=1.0).fingerprint()
