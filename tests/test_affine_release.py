import pytest
import torch

from fpavlm.affine_release import compile_affine_release
from fpavlm.cipher_native import (
    ClientCANREngine,
    DiagnosticBottleneckRelease,
    NoiseResilientMAViTEncoder,
)


def fixture_model(degree=1):
    torch.manual_seed(93)
    model = NoiseResilientMAViTEncoder(
        dim=4, num_layers=2, mlp_ratio=2, num_patches=4,
        in_channels=1, patch_size=2, img_size=4, num_regions=1,
        use_cadn=False, poly_act_degree=1, poly_norm_degree=degree,
        spectral_bound=3.0, ffn_spectral_bound=3.0,
        enable_noise_training=False, enable_token_mix=True,
        fuse_linear_poly_act=True, fuse_linear_poly_norm=degree == 1,
    ).eval()
    basis = torch.linalg.qr(torch.randn(4, 2))[0]
    release = DiagnosticBottleneckRelease(basis, torch.eye(4), token_groups=[[0, 1], [2, 3]])
    return model, release


def test_compilation_preserves_spatial_release_and_original_model():
    model, release = fixture_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    compiled = compile_affine_release(model, release)
    model.bake_statistics()
    patches = torch.randn(7, 4, 4)
    expected = release.encode_plain(model.forward_plain_fhe_match(patches))
    torch.testing.assert_close(compiled.plain(patches), expected, atol=2e-5, rtol=2e-5)
    for k, v in before.items():
        torch.testing.assert_close(model.state_dict()[k], v)
    assert compiled.multiplicative_depth == 1


def test_nonlinear_compilation_is_rejected():
    model, release = fixture_model(degree=2)
    with pytest.raises(ValueError, match="nonlinear"):
        compile_affine_release(model, release)
    model.use_cadn = True
    with pytest.raises(ValueError, match="non-CADN"):
        compile_affine_release(model, release)


def test_real_ckks_compiled_release_and_boundary():
    model, release = fixture_model()
    compiled = compile_affine_release(model, release)
    client = ClientCANREngine(poly_modulus_degree=8192, scale_bits=40, depth_budget=1,
                              generate_relin_keys=False)
    patches = torch.randn(1, 4, 4)
    encrypted, _ = client.encrypt_chunked(patches.flatten(), chunk_size=8)
    result = compiled.evaluate(encrypted, chunk_size=8)
    decrypted = client.decrypt(result, dim=1).reshape(1, *compiled.output_shape)
    torch.testing.assert_close(decrypted, compiled.plain(patches), atol=3e-4, rtol=3e-4)
    with pytest.raises(ValueError, match="count"):
        compiled.evaluate(encrypted[:-1], chunk_size=8)
    with pytest.raises(ValueError, match="secret key"):
        compiled.evaluate([client.ts.ckks_vector(client.context, [1.] * 8)] * 2, chunk_size=8)
