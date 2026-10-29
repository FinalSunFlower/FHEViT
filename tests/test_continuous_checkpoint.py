import tempfile
import unittest
from pathlib import Path

import torch

from fpavlm.continuous_checkpoint import (
    convert_cadn_checkpoint,
    validate_continuous_checkpoint,
)


class ContinuousCheckpointTests(unittest.TestCase):
    def test_candidate_is_explicitly_rejected(self):
        value = {
            "metadata": {
                "schema_version": 1,
                "profile": "model_confidential",
                "checkpoint_kind": "continuous_fhe",
                "architecture": {"use_cadn": False, "continuous_depth": 21},
                "validation": {"status": "candidate"},
            }
        }
        with self.assertRaisesRegex(ValueError, "not validated"):
            validate_continuous_checkpoint(value)

    def test_shallow_wide_variant_is_bound_to_checkpoint_metadata(self):
        value = {
            "metadata": {
                "schema_version": 1,
                "profile": "model_confidential",
                "checkpoint_kind": "continuous_fhe",
                "architecture": {
                    "use_cadn": False,
                    "continuous_depth": 6,
                    "architecture_variant": "canonical",
                },
                "validation": {"status": "candidate"},
            }
        }
        with self.assertRaisesRegex(ValueError, "architecture variant"):
            validate_continuous_checkpoint(
                value,
                require_validated=False,
                expected_architecture_variant="shallow_wide",
            )

    def test_conversion_has_strict_metadata_and_loadable_state(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            output = Path(directory) / "continuous.pt"
            # Minimal CADN-shaped artifact for the schema conversion test.
            enc = {
                "blocks.0.norm1.layer_norm.weight": torch.ones(2),
                "blocks.0.norm1.layer_norm.bias": torch.zeros(2),
                "blocks.0.norm1._ema_mean": torch.zeros(2),
                "blocks.0.norm1._ema_mean_x2": torch.ones(2),
                "final_norm.layer_norm.weight": torch.ones(2),
                "final_norm.layer_norm.bias": torch.zeros(2),
                "final_norm._ema_mean": torch.zeros(2),
                "final_norm._ema_mean_x2": torch.ones(2),
            }
            torch.save({"encoder_state_dict": enc, "adapter_state_dict": {}}, source)
            metadata = convert_cadn_checkpoint(
                source, output, num_layers=1, num_regions=1
            )
            self.assertEqual(metadata["validation"]["status"], "candidate")
            value = torch.load(output, map_location="cpu", weights_only=True)
            self.assertIn("blocks.0.norm1.c1", value["encoder_state_dict"])
            self.assertNotIn("blocks.0.norm1.layer_norm.weight", value["encoder_state_dict"])


if __name__ == "__main__":
    unittest.main()
