import unittest

import torch

from fpavlm.cipher_native.canr import ClientCANREngine, ServerCANREngine


class CipherBoundaryTests(unittest.TestCase):
    def test_chunked_raw_patch_encryption_preserves_public_boundary(self):
        client = ClientCANREngine(poly_modulus_degree=8192, scale_bits=20, depth_budget=2)
        chunks, layout = client.encrypt_chunked(
            torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32),
            chunk_size=2,
        )
        self.assertEqual(layout["parts_per_token"], 2)
        self.assertEqual(layout["feature_dim"], 4)
        for chunk in chunks:
            with self.assertRaises(ValueError):
                chunk.decrypt()
        recovered = client.decrypt(chunks, dim=2)
        self.assertTrue(torch.allclose(recovered, torch.tensor([[1.0, 2.0], [3.0, 4.0]]), atol=0.1))

    def test_uploaded_and_server_outputs_are_public_context_ciphertexts(self):
        client = ClientCANREngine(poly_modulus_degree=8192, scale_bits=20, depth_budget=2)
        server = ServerCANREngine(client)

        uploaded = client.encrypt(torch.tensor([[1.25, -2.5]], dtype=torch.float32))
        with self.assertRaises(ValueError):
            uploaded[0].decrypt()

        output = server.matmul_plain(uploaded, torch.eye(2))
        with self.assertRaises(ValueError):
            output[0].decrypt()

        recovered = client.decrypt(output, dim=2)
        self.assertTrue(torch.allclose(recovered, torch.tensor([[1.25, -2.5]]), atol=0.1))

        refreshed = client.canr_refresh(output)
        with self.assertRaises(ValueError):
            refreshed[0].decrypt()

    def test_degree_one_key_profile_can_omit_relinearization_keys(self):
        client = ClientCANREngine(
            poly_modulus_degree=8192,
            scale_bits=20,
            depth_budget=2,
            generate_relin_keys=False,
        )
        self.assertFalse(client.generate_relin_keys)
        server = ServerCANREngine(client)
        uploaded = client.encrypt(torch.tensor([[1.0, -1.0]]))
        output = server.matmul_plain(uploaded, torch.eye(2))
        recovered = client.decrypt(output, dim=2)
        self.assertTrue(torch.allclose(recovered, torch.tensor([[1.0, -1.0]]), atol=0.1))


if __name__ == "__main__":
    unittest.main()
