import torch

from fpavlm.cipher_native import ClientCANREngine
from fpavlm.spatial_quadratic import SpatialQuadraticRelease


def test_nonlinear_spatial_release_real_ckks():
    torch.manual_seed(180)
    model = SpatialQuadraticRelease(patch_dim=4, latent_dim=2, rank=2).eval()
    with torch.no_grad():
        model.square_weight.fill_(.2)
        model.symmetry_gain.fill_(.3)
    patches = torch.randn(1, 16, 4)
    # Nonzero second difference rules out accidentally testing an affine model.
    assert (model(patches) + model(-patches) - 2 * model(torch.zeros_like(patches))).norm() > .1
    circuit = model.compile()
    client = ClientCANREngine(poly_modulus_degree=8192, scale_bits=30, depth_budget=3)
    encrypted, _ = client.encrypt_chunked(patches.flatten(), chunk_size=32)
    output = circuit.evaluate(encrypted, chunk_size=32)
    decrypted = client.decrypt(output, dim=1).reshape(1, *circuit.output_shape)
    torch.testing.assert_close(decrypted, model(patches), atol=.015, rtol=.015)
    assert circuit.multiplicative_depth == 3
