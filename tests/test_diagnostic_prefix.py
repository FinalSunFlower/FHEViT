import hashlib

import pytest
import torch

from fpavlm.diagnostic_prefix import DiagnosticPrefixProjector, PrefixSpec
from fpavlm.spatial_quadratic import SpatialQuadraticRelease


def test_gaussian_ridge_and_binding(tmp_path):
    torch.manual_seed(283)
    spec = PrefixSpec("fixture", (2, 2), 3, 5, "a" * 64, "b" * 64)
    x = torch.randn(60, 2, 2)
    w = torch.randn(4, 15)
    y = (x.flatten(1) @ w + .5).reshape(60, 3, 5)
    model = DiagnosticPrefixProjector(spec).fit(x, y, sigma=0., ridge=.001)
    torch.testing.assert_close(model(x), y, rtol=1e-3, atol=1e-3)
    regularized = DiagnosticPrefixProjector(spec).fit(x, y, sigma=10., ridge=.001)
    assert regularized.weight.norm() < model.weight.norm() / 20
    p = tmp_path / "prefix.pt"
    model.save(str(p), training_patient_ids=["training_patient"])
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    loaded = DiagnosticPrefixProjector.load(str(p), expected_sha256=digest, expected_spec=spec)
    torch.testing.assert_close(loaded(x), model(x))
    wrong = PrefixSpec("other", (2, 2), 3, 5, "a" * 64, "b" * 64)
    with pytest.raises(ValueError, match="binding"):
        DiagnosticPrefixProjector.load(str(p), expected_sha256=digest, expected_spec=wrong)


def test_diagnostic_initialization_improves_training_fit():
    torch.manual_seed(7)
    patches = torch.randn(30, 16, 12)
    teacher = SpatialQuadraticRelease(12, 3, 2)
    targets = teacher(patches).detach()
    student = SpatialQuadraticRelease(12, 3, 2)
    before = (student(patches) - targets).square().mean()
    student.initialize_from_diagnostic_targets(patches, targets, ridge=.01)
    after = (student(patches) - targets).square().mean()
    assert after < before / 2


@pytest.mark.parametrize("shape,tokens", [((2,), 3), ((2, True), 3), ((2, 2), 3.5), ((2, 0), 3)])
def test_prefix_rejects_invalid_dimensions(shape, tokens):
    with pytest.raises(ValueError, match="dimensions"):
        PrefixSpec("fixture", shape, tokens, 5, "a" * 64, "b" * 64)
