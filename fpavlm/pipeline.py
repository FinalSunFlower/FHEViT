"""High-level plug-and-play API for FHE-protected medical VLM inference.

Provides a typed orchestration API for encrypted vision, governed embedding
release, per-LLM visual adaptation, and medical text generation.

DESIGN GOALS:
  1. One-line API: `report = pipeline.run(image_path)` (hides FHE complexity)
  2. Self-contained: user doesn't need to understand CFP/CADN/CANR internals
  3. Extensible: swap LLM via LLMRegistry, swap FHE params via config
  4. Research-prototype orchestration: handles model loading and FHE context setup

ARCHITECTURE:
  User → FHEViTPipeline.run(image_path) →
    [Client] patch_embed → encrypt → upload ciphers
    [Server] FHE ViT (CADN + PolyAct + LTM) → W_concept → 16 ciphers
    [Client] CANR/CADN refresh ×5 → decrypt → DPCFP → noised embedding
    [Server] Perceiver (16→256) → LLM (MedGemma 4B) → report
    [Client] receive report

USAGE (minimal quickstart):
    from fpavlm import FHEViTPipeline

    pipeline = FHEViTPipeline.from_pretrained(
        llm_name="medgemma-4b",
        llm_model_path="models/unsloth/medgemma-4b-it",
        encoder_weights="artifacts/encoder_weights.pt",
        medical_priors="artifacts/medical_priors.pt",
        cfp_sensitivity="artifacts/cfp_sensitivity.pt",
    )
    report = pipeline.run("data/openi_subset/images/openi_00001.png")
    print(report)

USAGE (with explicit config):
    from fpavlm import FHEViTPipeline, FHEConfig

    config = FHEConfig(
        ckks_poly_modulus_degree=32768,
        ckks_scale_bits=28,
        ckks_depth_budget=13,
        use_cadn=True,
        cfp_noise_fraction=24.0,
        cfp_clip_norm=1.0,
        cfp_delta=1e-5,
        cfp_quantize=True,
        llm_num_beams=4,
        llm_seed=42,
        use_mbr_decoding=True,
        mbr_k=4,
    )
    pipeline = FHEViTPipeline.from_pretrained(..., config=config)
    report = pipeline.run("image.png")

PRIVACY GUARANTEE:
    The pipeline returns a DPReport alongside the report. With the corrected
    per-coordinate Gaussian accounting, the default public noise scale gives
    ε≈223.27 at δ=1e-5, which is not a meaningful strong-privacy guarantee.
    (Computed by `DPCFPAccountant` in `fpavlm/cipher_native/dp_accountant.py`,
    using the Rényi DP accountant over the default alpha grid.)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from fpavlm.anatomical_slots import canonical_anatomical_slots, slot_token_groups
from fpavlm.checkpoint import load_artifact_manifest, load_tensor_checkpoint

# Local imports (project-internal)
from fpavlm.cipher_native import (
    AnatomyAwareTokenRouter,
    CiphertextTensor,
    ClientCANREngine,
    NoiseResilientMAViTEncoder,
    ServerCANREngine,
)
from fpavlm.cipher_native.dp_accountant import DPCFPAccountant, DPReport
from fpavlm.continuous_checkpoint import validate_continuous_checkpoint
from fpavlm.deployment import (
    DeploymentProfile,
    continuous_fhe_depth,
    validate_ckks_budget,
)
from fpavlm.embedding_contract import DiagnosticEmbeddingRelease, LLMVisualEmbedding
from fpavlm.llm import LLMInterface, LLMRegistry
from fpavlm.llm.noise_aware_decoding import (
    cnad_generate,
    noise_aware_generate,
)
from fpavlm.noise_calibration import FHENoiseCalibration
from fpavlm.preprocessing import (
    ImagePreprocessingSpec,
)
from fpavlm.preprocessing import (
    preprocess_image as canonical_preprocess_image,
)
from fpavlm.privacy_audit import validate_release_calibration
from fpavlm.vision_plugin import FHEVisionCapabilities, FHEVisionRegistry

MEDICAL_POLYNOMIAL_CAPABILITIES = FHEVisionCapabilities(
    input_domain="both",
    output_domain="diagnostic_coefficients",
    homomorphic_operations=(
        "add",
        "cipher_plain_mul",
        "cipher_cipher_mul",
    ),
    supports_interactive=True,
    supports_callback_free=True,
    protects_server_weights_callback_free=True,
    supports_raw_patch_ciphertext=True,
)


@FHEVisionRegistry.register(
    "medical-polynomial-vit",
    capabilities=MEDICAL_POLYNOMIAL_CAPABILITIES,
)
def _build_medical_polynomial_vit(**kwargs):
    return NoiseResilientMAViTEncoder(**kwargs)

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FHEConfig:
    """Configuration for FHE-VLM pipeline.

    Defaults provide a research baseline. Deployment profiles require
    explicit privacy, integrity, and CKKS-budget choices.
    """
    # Explicit trust boundary. The baseline reproduces historical experiments;
    # deployment profiles enforce stronger, mutually exclusive contracts.
    deployment_profile: str = DeploymentProfile.RESEARCH_BASELINE.value
    vision_plugin_name: str = "medical-polynomial-vit"

    # CKKS parameters (interactive baseline: N=32768, scale=2^28, depth 13)
    ckks_poly_modulus_degree: int = 32768
    ckks_scale_bits: int = 28
    ckks_depth_budget: int = 13

    # Encoder architecture
    dim: int = 64
    num_layers: int = 3
    num_patches: int = 16
    grid_size: int = 4
    patch_size: int = 56
    image_size: int = 224  # 4 * 56
    in_channels: int = 3
    mlp_ratio: float = 2.0
    spectral_bound: float = 3.0
    input_range: float = 3.0
    poly_act_degree_schedule: List[int] = field(default_factory=lambda: [1, 1, 1])
    # Degree of the server-side PolyNorm. Secure callback-free deployments use
    # degree 1 so the complete circuit fits TenSEAL's N=32768 prime limit.
    poly_norm_degree: Optional[int] = None
    num_regions: int = 5
    llm_hidden_dim: int = 2560
    num_concepts: int = 32
    student_grid: int = 4
    target_grid: int = 16
    perceiver_config: Dict = field(default_factory=lambda: {
        "num_heads": 4, "attention_dim": 512, "ffn_dim": 2560,
        "num_layers": 1, "init_from_bilinear": True,
    })
    enable_token_mix: bool = True
    token_mix_spectral_bound: float = 3.0
    # Algebraic callback-free optimization.  When enabled with degree-1
    # PolyAct, folds the affine activation into W_o and removes one CKKS level
    # per block.  Disabled by default for checkpoint/trace compatibility.
    fuse_linear_poly_act: bool = False
    fuse_linear_poly_norm: bool = False
    channelwise_polynorm: bool = False
    fuse_final_affine: bool = False

    # Interactive client-assisted dynamic normalization.
    use_cadn: bool = True

    # Optional pre-CADN Gaussian masking for empirical model-extraction studies.
    # Heuristic masking only: when > 0, the server adds Gaussian masking noise
    # N(0, sigma^2) to each exhausted ciphertext in the encrypted domain (cipher+
    # plain, 0 depth) BEFORE sending it to the client for CADN refresh. Defends
    # against the model-stealing attack where the client recovers W from
    # (X, X·W) pairs observed during CADN.
    # Without defense: recovery error = 5e-14 (exact recovery with 64 queries).
    # With sigma=0.01:  recovery error = 0.92 @ N=64 (attack fails), cos >= 0.9999.
    # This does not provide cryptographic model confidentiality. Use the
    # callback-free model_confidential profile for that threat model.
    # Default 0.01 reproduces the evaluated baseline. Set 0.0 to disable for ablation.
    pre_cadn_noise_sigma: float = 0.01

    # Select a registered private server-side perceiver.
    # "v162-default" is retained as a stable registry identifier.
    # "identity" = IdentityPerceiver (no-op, repeat 16×16=256, 0 params, ablation)
    # Future: "anatomy-aware", "frequency-aware", etc. can be registered via
    # @PerceiverRegistry.register("name")
    perceiver_name: str = "v162-default"

    # Medical-domain scale selection.
    # When True (default), FHEConfig.__post_init__ calls mafo_select_optimal_scale
    # to compute the optimal CKKS scale bits from medical feature statistics
    # (kurtosis, max_norm). Overrides ckks_scale_bits with the computed value.
    # Saves ~4 bits per modulus factor → ~14% smaller ciphertexts → ~14% faster.
    # When False, uses the hardcoded ckks_scale_bits (default 28, v165 canonical).
    use_mafo_scale: bool = True
    # Medical feature statistics for MAFO-C scale selection.
    # Empirically measured on OpenI chest X-ray features (dim=64 after patch_proj).
    # Users can override with their own dataset's statistics.
    mafo_feature_stats: dict = None  # default set in __post_init__

    # Client-governed record-level Gaussian release.
    cfp_noise_fraction: float = 24.0  # v164 canonical anisotropic σ
    cfp_clip_norm: float = 1.0        # L2 clip for bounded sensitivity
    cfp_delta: float = 1e-5           # DP delta (standard for medical)
    cfp_quantize: bool = True         # int8 extra defense + bandwidth
    cfp_anisotropy_lambda: float = 1.0  # v164 Innovation B
    cfp_target_epsilon: Optional[float] = None
    daar_axis_sensitivity: Optional[list] = None
    cfp_renormalize_sensitivity: bool = True

    # LLM generation
    llm_num_beams: int = 4
    llm_seed: int = 42
    llm_max_new_tokens: int = 200

    # Optional noise-aware decoding.
    use_mbr_decoding: bool = False
    mbr_k: int = 4
    mbr_beam_width: int = 2
    # CNAD/MBR require paired FHE-vs-plaintext release calibration. A fresh
    # re-encryption differential measures only a lower bound and is not accepted.
    noise_calibration_path: Optional[str] = None

    # CNAD (Cipher-Noise-Aware Decoding) for measured FHE noise.
    # When True (default), pipeline uses cnad_generate() which composes three
    # FHE-specific mechanisms on top of MBR + Top-nσ:
    #   1. NAT (Noise-Adaptive Temperature) — logit scaling by 1/T where T is
    #      derived from σ_fhe (compatible with beam search).
    #   2. NCT (Noise-Calibrated Top-σ) — n = n_base + k_n × (σ_fhe / σ_max).
    #   3. NPM (Noise-Proportional MBR) — perturbs using ACTUAL σ_fhe.
    # When False, falls back to noise_aware_generate() (v168 generic NLP decoding).
    use_cnad_decoding: bool = False

    # CADN requires the configured interactive refresh schedule; no hidden
    # noise-budget shortcut is applied.
    # with CADN path (CADN requires per-block decrypt for z_norm, so CANR cannot
    # be skipped). The DEPLOYED FHE latency optimization is CADN itself (per-block
    # depth 5→4, 20% reduction).

    # Lazy Block Refresh (LBR) — latency-accuracy trade-off.
    # When lbr_enabled=True, the pipeline uses LazyBlockRefreshScheduler to
    # reduce CADN round trips from 5 to ~2 (60 % reduction).
    # Strategies: "full_cadn" / "diagnostic_priority" / "min_refreshes" / "uniform"
    # Expected speedup (3 blocks, fraction=0.5): ~2.2× on round-trip cost.
    # Accuracy: cos_16 ≈ 0.9995 vs 1.000000 for full CADN (negligible for generation).
    # Only relevant for the interactive (CADN) path; ignored for model_confidential.
    lbr_enabled: bool = False
    lbr_strategy: str = "diagnostic_priority"
    lbr_cadn_fraction: float = 0.5

    # Position-Based Selective Refresh (PBSR) — per-token refresh selectivity.
    # When pbsr_enabled=True, only anatomically diagnostic tokens (default 10/16)
    # receive CADN; background tokens identity-skip every multiply.
    # Speedup alone: ~1.6×; combined LBR+PBSR: ~3.5× (refresh ops: 80→23).
    pbsr_enabled: bool = False
    pbsr_coverage: float = 0.625   # 10/16 = 62.5 % for default 4×4 CXR grid

    # Optional trained LLM embedding adapter.
    # When True (default), pipeline auto-applies LLMEmbeddingAdapter when the LLM's
    # hidden_dim != encoder output dim (2560). Uses PCA-based zero-shot projection
    # + native feature distribution calibration. Enables true plug-and-play LLM
    # swapping (MedGemma/InternVL2/Qwen2-VL) without retraining the FHE encoder.
    use_llm_embedding_adapter: bool = True
    allow_zero_shot_adapter: bool = False
    allow_remote_model_code: bool = False
    trusted_remote_code_sha256: Optional[str] = None
    artifact_manifest_path: Optional[str] = None

    # Configurable adapters directory.
    # When unset, the repository layout uses artifacts/adapters. Installed
    # users should set an explicit absolute path.
    adapters_dir: Optional[str] = None

    # Diagnostic Subspace Retention release bottleneck.
    # When set, the pipeline loads a DSR calibration (.pt) and folds the
    # projection Π into W_concept after loading encoder weights. It retains a
    # diagnosis-predictive subspace but does not guarantee de-identification.
    # Zero additional FHE depth (folded into existing cipher×plain matmul).
    # Default None (DSR disabled). Set to a .pt path to enable.
    dsr_calibration_path: Optional[str] = None

    # Optional linear probe over the governed FDBR coefficients. It does not
    # change the release; it only conditions the report prompt on labels that
    # are already functions of the decrypted diagnostic statistic.
    finding_probe_path: Optional[str] = None
    # Research-only: load a failed development checkpoint for ablation.
    # Secure deployment profiles still require a feasible audit.
    allow_infeasible_release: bool = False

    # The baseline remains usable but makes no strong release-privacy claim.
    # Deployment profiles require a feasible FDBR calibration.
    release_mode: str = "legacy_embedding"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Circuit/release presets are explicit because a legacy checkpoint cannot be
    # safely relabelled as a new mechanism. ``shallow_wide`` selects a two-block
    # callback-free circuit when constructing a fresh model; ``anatomical``
    # requires the calibration artifact to contain the canonical five-slot map.
    architecture_variant: str = "canonical"
    release_slot_mode: str = "calibration"
    anatomical_slot_include_background: bool = True
    anatomical_slot_rank: int = 2
    use_anatomical_prompt_anchors: bool = True

    @classmethod
    def shallow_wide(cls, **kwargs) -> "FHEConfig":
        """Return a conservative shallow/wide callback-free configuration.

        The preset changes only circuit shape and polynomial schedule; callers
        still need a checkpoint trained with the same values and a matching
        release calibration.
        """
        kwargs.setdefault("architecture_variant", "shallow_wide")
        kwargs.setdefault("num_layers", 2)
        kwargs.setdefault("poly_act_degree_schedule", [1, 1])
        kwargs.setdefault("poly_norm_degree", 1)
        kwargs.setdefault("enable_token_mix", True)
        kwargs.setdefault("fuse_linear_poly_act", True)
        kwargs.setdefault("fuse_linear_poly_norm", True)
        kwargs.setdefault("fuse_final_affine", True)
        kwargs.setdefault("use_cadn", False)
        return cls(**kwargs)

    def anatomical_release_groups(self) -> List[List[int]]:
        """Return the canonical groups expected by an anatomical calibration."""
        return slot_token_groups(
            num_tokens=self.num_patches,
            grid_size=self.grid_size,
            include_background=self.anatomical_slot_include_background,
        )

    def __post_init__(self):
        try:
            profile = DeploymentProfile(self.deployment_profile)
        except ValueError as exc:
            allowed = ", ".join(profile.value for profile in DeploymentProfile)
            raise ValueError(f"deployment_profile must be one of: {allowed}") from exc
        FHEVisionRegistry.capabilities(self.vision_plugin_name).validate_profile(profile)
        if self.architecture_variant not in {"canonical", "shallow_wide"}:
            raise ValueError("architecture_variant must be 'canonical' or 'shallow_wide'")
        if self.release_slot_mode not in {"calibration", "anatomical"}:
            raise ValueError("release_slot_mode must be 'calibration' or 'anatomical'")
        if isinstance(self.anatomical_slot_rank, bool) or self.anatomical_slot_rank < 1:
            raise ValueError("anatomical_slot_rank must be a positive integer")
        if self.anatomical_slot_rank > self.dim:
            raise ValueError("anatomical_slot_rank cannot exceed encoder dimension")
        if self.architecture_variant == "shallow_wide":
            if self.num_layers > 2:
                raise ValueError("shallow_wide architecture requires num_layers <= 2")
            schedule = self.poly_act_degree_schedule
            if isinstance(schedule, list) and len(schedule) != self.num_layers:
                raise ValueError("poly_act_degree_schedule must match num_layers")
            if schedule not in (None, "auto") and any(degree != 1 for degree in schedule):
                raise ValueError("shallow_wide architecture requires degree-1 PolyAct")
            if self.use_cadn:
                raise ValueError("shallow_wide callback-free architecture requires use_cadn=False")
        if self.image_size != self.grid_size * self.patch_size:
            raise ValueError(
                f"image_size {self.image_size} != grid_size {self.grid_size} * patch_size {self.patch_size}"
            )
        if self.poly_norm_degree is not None and self.poly_norm_degree not in (1, 2):
            raise ValueError("poly_norm_degree must be 1 or 2")
        if self.release_mode not in ("diagnostic_bottleneck", "legacy_embedding"):
            raise ValueError(
                "release_mode must be 'diagnostic_bottleneck' or 'legacy_embedding'"
            )
        if self.release_mode == "diagnostic_bottleneck" and not self.dsr_calibration_path:
            raise ValueError(
                "diagnostic_bottleneck release requires dsr_calibration_path"
            )
        if self.release_slot_mode == "anatomical" and self.num_patches != self.grid_size * self.grid_size:
            raise ValueError("anatomical release requires a square patch grid")
        if self.release_slot_mode == "anatomical" and self.release_mode != "diagnostic_bottleneck":
            raise ValueError("anatomical release requires release_mode='diagnostic_bottleneck'")

        if profile is not DeploymentProfile.RESEARCH_BASELINE:
            if self.release_mode != "diagnostic_bottleneck":
                raise ValueError(
                    f"{profile.value} requires release_mode='diagnostic_bottleneck'"
                )
            if self.cfp_target_epsilon is None:
                raise ValueError(
                    f"{profile.value} requires an explicit cfp_target_epsilon"
                )
            if self.allow_zero_shot_adapter:
                raise ValueError("deployment profiles forbid zero-shot adapter fallback")
            if not self.artifact_manifest_path:
                raise ValueError(
                    f"{profile.value} requires artifact_manifest_path for checkpoint integrity"
                )
        if profile is DeploymentProfile.UTILITY_INTERACTIVE and not self.use_cadn:
            raise ValueError("utility_interactive requires use_cadn=True")
        if profile is DeploymentProfile.MODEL_CONFIDENTIAL:
            if self.use_cadn:
                raise ValueError(
                    "model_confidential requires use_cadn=False and a separately "
                    "trained polynomial-normalization checkpoint"
                )
            if self.pre_cadn_noise_sigma != 0:
                raise ValueError(
                    "model_confidential has no CADN boundary; pre_cadn_noise_sigma must be 0"
                )
            if self.poly_norm_degree not in (None, 1, 2):
                raise ValueError(
                    "model_confidential supports poly_norm_degree 1 or 2"
                )
            if self.poly_norm_degree is None:
                self.poly_norm_degree = 1
        elif self.poly_norm_degree is None:
            self.poly_norm_degree = 2
        if (self.use_cnad_decoding or self.use_mbr_decoding) and not self.noise_calibration_path:
            raise ValueError(
                "noise-aware decoding requires noise_calibration_path from paired "
                "FHE/plain release measurements"
            )

        # v167 MAFO-C: Medical-Domain-Informed Scale Selection
        # When enabled, compute optimal CKKS scale bits from medical feature
        # statistics (kurtosis, max_norm). Saves ~4 bits per modulus factor.
        if self.use_mafo_scale:
            # Default medical feature statistics (OpenI chest X-ray, dim=64)
            # Empirically measured. Users can override via mafo_feature_stats.
            if self.mafo_feature_stats is None:
                self.mafo_feature_stats = {
                    'max_norm': 5.2,         # Max L2 norm across features
                    'mean_norm': 2.8,        # Mean L2 norm
                    'std_norm': 1.1,         # Std of L2 norms
                    'kurtosis': 10.0,        # Heavy-tailed (vs Gaussian 3.0)
                    'depth': self.ckks_depth_budget,
                }
            # Lazy import to avoid circular dependency at module load
            from .cipher_native.mafo import mafo_select_optimal_scale
            mafo_result = mafo_select_optimal_scale(
                self.mafo_feature_stats,
                target_cos=0.9999,
                safety_factor=2.0,
            )
            # Override ckks_scale_bits with MAFO-computed value
            # (only if MAFO recommends a SMALLER scale — never increase,
            # to avoid breaking the 128-bit security bound)
            mafo_scale_bits = mafo_result['scale_bits']
            if mafo_scale_bits < self.ckks_scale_bits:
                self._mafo_scale_rationale = mafo_result['rationale']
                self._mafo_scale_reduction_pct = mafo_result['ciphertext_size_reduction_pct']
                self.ckks_scale_bits = mafo_scale_bits
            else:
                # MAFO recommends same or larger scale — keep default
                self._mafo_scale_rationale = (
                    f"MAFO-C: recommended scale 2^{mafo_scale_bits} >= default "
                    f"2^{self.ckks_scale_bits}, keeping default (no benefit)."
                )
                self._mafo_scale_reduction_pct = 0.0

        required_depth = self.ckks_depth_budget
        if (
            profile is DeploymentProfile.MODEL_CONFIDENTIAL
            or self.architecture_variant == "shallow_wide"
        ):
            required_depth = continuous_fhe_depth(
                self.poly_act_degree_schedule,
                token_mixing=self.enable_token_mix,
                poly_norm_degree=self.poly_norm_degree,
                fuse_linear_poly_act=self.fuse_linear_poly_act,
                fuse_linear_poly_norm=self.fuse_linear_poly_norm,
                fuse_final_affine=self.fuse_final_affine,
            )
            if self.ckks_depth_budget < required_depth:
                raise ValueError(
                    f"model_confidential requires ckks_depth_budget >= {required_depth}; "
                    f"got {self.ckks_depth_budget}"
                )
        self._ckks_budget = validate_ckks_budget(
            self.ckks_poly_modulus_degree,
            self.ckks_scale_bits,
            self.ckks_depth_budget,
        )
        self._required_continuous_depth = required_depth

        # LBR schedule (pre-computed; stored for pipeline use)
        if self.lbr_enabled and self.use_cadn:
            from .cipher_native.lbr_scheduler import default_lbr_schedule
            self._lbr_schedule = default_lbr_schedule(
                num_blocks=self.num_layers,
                use_cadn=True,
                poly_norm_degree=self.poly_norm_degree,
                enable_token_mix=self.enable_token_mix,
                depth_budget=self.ckks_depth_budget,
                strategy=self.lbr_strategy,
                cadn_fraction=self.lbr_cadn_fraction,
            )
        else:
            self._lbr_schedule = None


# =============================================================================
# Pipeline
# =============================================================================

class FHEViTPipeline:
    """End-to-end FHE-VLM pipeline for privacy-preserving medical report generation.

    One-line API: `report = pipeline.run(image_path)`.

    The pipeline handles:
      1. Image preprocessing (resize, patchify, normalize)
      2. Client-side FHE encryption (real CKKS)
      3. Server-side FHE ViT inference (CADN + PolyAct + LTM)
      4. Client-side CANR/CADN refresh + decrypt
      5. Client-side DPCFP (accounted embedding perturbation)
      6. Server-side Perceiver + LLM (MedGemma 4B)
      7. Noise-aware MBR decoding (robust to FHE noise)

    Privacy contract:
      - Server NEVER sees raw image (only encrypted ciphers + noised embedding)
      - Client receives public patch-projection and positional-embedding parameters
      - Released plaintext embedding has an explicit DP report; the default ε is weak
    """

    def __init__(
        self,
        encoder: NoiseResilientMAViTEncoder,
        llm: LLMInterface,
        client_engine: ClientCANREngine,
        server_engine: ServerCANREngine,
        medical_priors: Dict[str, torch.Tensor],
        cfp_sensitivity: Optional[torch.Tensor] = None,
        config: Optional[FHEConfig] = None,
    ):
        self.encoder = encoder
        self.llm = llm
        self.client_engine = client_engine
        self.server_engine = server_engine
        self.medical_priors = medical_priors
        self.cfp_sensitivity = cfp_sensitivity
        self.config = config or FHEConfig()
        self.deployment_profile = DeploymentProfile(self.config.deployment_profile)
        self.finding_probe = None
        self.noise_calibration = None
        if self.config.noise_calibration_path:
            self.noise_calibration = FHENoiseCalibration.load(
                self.config.noise_calibration_path
            )

        # Build the accountant from the actual release boundary. A calibrated
        # bottleneck may pool 16 visual tokens into anatomical or global groups;
        # the accountant must never assume the encoder's input token count.
        bottleneck = self.encoder.concept_adapter.diagnostic_bottleneck
        release_dim = bottleneck.release_dim if bottleneck is not None else self.config.llm_hidden_dim
        release_tokens = (
            len(bottleneck.token_groups)
            if bottleneck is not None and bottleneck.token_groups is not None
            else self.config.num_patches
        )
        if bottleneck is not None and self.config.daar_axis_sensitivity:
            release_sensitivity = torch.tensor(
                self.config.daar_axis_sensitivity, dtype=torch.float32
            )
        elif bottleneck is None:
            release_sensitivity = cfp_sensitivity
        else:
            release_sensitivity = None
        self.dp_accountant = DPCFPAccountant(
            noise_fraction=self.config.cfp_noise_fraction,
            dim=release_tokens * release_dim,
            clip_norm=self.config.cfp_clip_norm,
            delta=self.config.cfp_delta,
            quantize=self.config.cfp_quantize,
            anisotropy_lambda=self.config.cfp_anisotropy_lambda,
            diagnostic_sensitivity=release_sensitivity,
            target_epsilon=self.config.cfp_target_epsilon,
            renormalize_sensitivity=self.config.cfp_renormalize_sensitivity,
        )

        # Warmup: bake encoder statistics (PolyNorm / CADN affine)
        self.encoder.eval()
        self.encoder.bake_statistics()

    @classmethod
    def from_pretrained(
        cls,
        llm_name: str = "medgemma-4b",
        llm_model_path: str = "models/unsloth/medgemma-4b-it",
        encoder_weights: str = "artifacts/encoder_weights.pt",
        medical_priors: str = "artifacts/medical_priors.pt",
        cfp_sensitivity: Optional[str] = "artifacts/cfp_sensitivity.pt",
        config: Optional[FHEConfig] = None,
    ) -> "FHEViTPipeline":
        """Construct pipeline from pretrained checkpoints (recommended entry point).

        Args:
            llm_name: LLM registry name (e.g., "medgemma-4b").
            llm_model_path: Path to local LLM weights.
            encoder_weights: Path to the encoder + adapter checkpoint.
            medical_priors: Path to medical priors (.pt).
            cfp_sensitivity: Path to anisotropic CFP sensitivity (.pt).
            config: Optional FHEConfig override.

        Returns:
            FHEViTPipeline ready to call .run().
        """
        cfg = config or FHEConfig()
        if (
            llm_name is None
            and DeploymentProfile(cfg.deployment_profile)
            is not DeploymentProfile.RESEARCH_BASELINE
        ):
            raise ValueError(
                "deployment profiles require a server-side LLM; use llm_name=None "
                "only for research encoder-only measurements"
            )
        artifact_hashes = (
            load_artifact_manifest(cfg.artifact_manifest_path)
            if cfg.artifact_manifest_path
            else {}
        )

        def checkpoint_hash(path: str) -> Optional[str]:
            normalized = os.path.normcase(os.path.abspath(os.fspath(path)))
            if normalized in artifact_hashes:
                return artifact_hashes[normalized]
            if artifact_hashes:
                raise ValueError(f"Artifact {path!r} is absent from the configured manifest")
            return None

        # Step 1: Load medical priors
        priors = load_tensor_checkpoint(
            medical_priors,
            required_keys=(
                "medical_prior", "anatomical_prior", "disease_prior_embeddings",
                "med_poly_act_prior", "disease_anchor_directions", "region_salience",
            ),
            expected_sha256=checkpoint_hash(medical_priors),
        )

        # Step 2: Build anatomy router (for symmetry pairs)
        router = AnatomyAwareTokenRouter(
            dim=1, num_tokens=cfg.num_patches, grid_size=cfg.grid_size,
            num_regions=cfg.num_regions,
        )
        symmetry_pairs = router.get_symmetry_pairs()

        # Step 3: Build encoder
        encoder = FHEVisionRegistry.load(
            cfg.vision_plugin_name,
            deployment_profile=cfg.deployment_profile,
            dim=cfg.dim, num_layers=cfg.num_layers,
            mlp_ratio=cfg.mlp_ratio, num_patches=cfg.num_patches,
            in_channels=cfg.in_channels, patch_size=cfg.patch_size,
            img_size=cfg.image_size, spectral_bound=cfg.spectral_bound,
            input_range=cfg.input_range,
            block_type="med_fhe_vit_v3", enable_noise_training=False,
            poly_act_degree_schedule=cfg.poly_act_degree_schedule,
            poly_norm_degree=cfg.poly_norm_degree,
            num_regions=cfg.num_regions, llm_dim=cfg.llm_hidden_dim,
            num_concepts=cfg.num_concepts, pool_type="perceiver_decoder",
            student_grid=cfg.student_grid, target_grid=cfg.target_grid,
            perceiver_config=cfg.perceiver_config,
            enable_token_mix=cfg.enable_token_mix,
            token_mix_spectral_bound=cfg.token_mix_spectral_bound,
             fuse_linear_poly_act=cfg.fuse_linear_poly_act,
             fuse_linear_poly_norm=cfg.fuse_linear_poly_norm,
             channelwise_polynorm=cfg.channelwise_polynorm,
             fuse_final_affine=cfg.fuse_final_affine,
            use_cadn=cfg.use_cadn,
            # v167: plug-and-play perceiver — passes perceiver_name to adapter
            # Default "v162-default" = PerceiverDecoder (backward compat)
            # "identity" = IdentityPerceiver (ablation)
            perceiver_name=cfg.perceiver_name,
            medical_prior=priors["medical_prior"],
            anatomical_prior=priors["anatomical_prior"],
            disease_prior_embeddings=priors["disease_prior_embeddings"],
            med_poly_act_prior=priors["med_poly_act_prior"],
            symmetry_pairs=symmetry_pairs,
            disease_anchor_directions=priors["disease_anchor_directions"],
            disease_anchor_alpha=0.3,
            region_salience=priors["region_salience"],
        )
        FHEVisionRegistry.validate_instance(
            encoder,
            profile=DeploymentProfile(cfg.deployment_profile),
            expected_continuous_depth=(
                cfg._required_continuous_depth
                if DeploymentProfile(cfg.deployment_profile)
                is DeploymentProfile.MODEL_CONFIDENTIAL
                else None
            ),
        )

        # Step 4: Load encoder weights
        checkpoint = load_tensor_checkpoint(
            encoder_weights,
            required_keys=("encoder_state_dict", "adapter_state_dict"),
            expected_sha256=checkpoint_hash(encoder_weights),
        )
        if DeploymentProfile(cfg.deployment_profile) is DeploymentProfile.MODEL_CONFIDENTIAL:
            validate_continuous_checkpoint(
                checkpoint,
                require_validated=True,
                expected_depth=cfg._required_continuous_depth,
                expected_preprocessing_fingerprint=ImagePreprocessingSpec(
                    image_size=cfg.image_size,
                    input_range=cfg.input_range,
                ).fingerprint(),
                expected_linear_polynorm_fusion=cfg.fuse_linear_poly_norm,
                expected_channelwise_polynorm=cfg.channelwise_polynorm,
                expected_architecture_variant=cfg.architecture_variant,
            )
        encoder_state = checkpoint["encoder_state_dict"]
        adapter_state = checkpoint["adapter_state_dict"]
        combined_state = {}
        for k, v in encoder_state.items():
            combined_state[k] = v
        for k, v in adapter_state.items():
            combined_state[f"concept_adapter.{k}"] = v
        encoder.load_state_dict(combined_state, strict=True)
        encoder.eval()
        encoder.bake_statistics()

        # Step 4b: load and validate the diagnostic release bottleneck.
        # Fold the projection Π into W_concept AFTER loading weights (so the
        # loaded W_concept gets DSR applied on top). Zero FHE depth change.
        # Shared ViT-space calibration; utility still needs per-LLM validation.
        if cfg.release_mode == "diagnostic_bottleneck":
            if not cfg.dsr_calibration_path:
                raise ValueError(
                    "diagnostic_bottleneck release requires dsr_calibration_path"
                )
            from .cipher_native.subspace_defense import DiagnosticSubspaceRetention
            dsr = DiagnosticSubspaceRetention.load(
                cfg.dsr_calibration_path,
                expected_sha256=checkpoint_hash(cfg.dsr_calibration_path),
            )
            validate_release_calibration(
                dsr.metrics,
                require_privacy_audit=(
                    DeploymentProfile(cfg.deployment_profile)
                    is not DeploymentProfile.RESEARCH_BASELINE
                ),
                allow_infeasible=bool(cfg.allow_infeasible_release)
                and DeploymentProfile(cfg.deployment_profile)
                is DeploymentProfile.RESEARCH_BASELINE,
                expected_mechanism=(
                    {
                        "accountant": "rdp_gaussian",
                        "adjacency": "replace_one_image_2R",
                        "whole_record_clip_norm": float(cfg.cfp_clip_norm),
                        "target_epsilon": float(cfg.cfp_target_epsilon),
                        "delta": float(cfg.cfp_delta),
                        # Pooling is part of the audited mechanism; do not
                        # infer it from the encoder's pre-pooling patch count.
                        "release_token_count": int(
                            dsr.metrics["mechanism"]["release_token_count"]
                        ),
                        "release_dim": int(dsr.k),
                        "quantized": bool(cfg.cfp_quantize),
                        "anisotropy_lambda": 0.0,
                        "preprocessing_fingerprint": ImagePreprocessingSpec(
                            image_size=cfg.image_size,
                            input_range=cfg.input_range,
                        ).fingerprint(),
                    }
                    if DeploymentProfile(cfg.deployment_profile)
                    is not DeploymentProfile.RESEARCH_BASELINE
                    else None
                ),
            )
            w_in = encoder.concept_adapter.W_concept.weight.shape[0]  # D_in
            if dsr.subspace.shape[0] != w_in:
                raise ValueError(
                    f"FDBR basis input dim {dsr.subspace.shape[0]} != encoder dim {w_in}"
                )
            mechanism = dsr.metrics.get("mechanism", {})
            if cfg.release_slot_mode == "anatomical":
                expected_groups = cfg.anatomical_release_groups()
                actual_groups = mechanism.get("token_groups")
                if actual_groups != expected_groups:
                    raise ValueError(
                        "anatomical release requires a calibration with the "
                        f"canonical {len(expected_groups)}-slot partition"
                    )
                if int(mechanism.get("release_dim", dsr.k)) != int(cfg.anatomical_slot_rank):
                    raise ValueError(
                        "anatomical release rank does not match calibration: "
                        f"expected {cfg.anatomical_slot_rank}, got {mechanism.get('release_dim')}"
                    )
            if mechanism.get("daar_enabled"):
                cfg.cfp_anisotropy_lambda = float(mechanism.get("daar_lambda", 0.0))
                cfg.daar_axis_sensitivity = mechanism.get("daar_axis_sensitivity")
                cfg.cfp_renormalize_sensitivity = bool(
                    mechanism.get("daar_renormalize_sensitivity", False)
                )
            encoder.concept_adapter.configure_diagnostic_bottleneck(
                dsr.subspace,
                token_groups=mechanism.get("token_groups"),
            )
            encoder.concept_adapter.configure_public_spatial_prior(
                encoder.pos_embed[:, : encoder.concept_adapter.num_tokens, :]
            )
            encoder.concept_adapter.prepare_cipher_weights()
            m = dsr.metrics
            details = [f"k={dsr.k}/{w_in}"]
            if "retained_feature_variance" in m:
                details.append(
                    f"retained_var={float(m['retained_feature_variance']):.4f}"
                )
            if "diagnosis_explained_variance" in m:
                details.append(
                    f"diag_explained={float(m['diagnosis_explained_variance']):.4f}"
                )
            selected = m.get("selected_candidate")
            if isinstance(selected, dict) and "utility_ap" in selected:
                details.append(f"utility_ap={float(selected['utility_ap']):.4f}")
            evidence = m.get("evidence_scope")
            if isinstance(evidence, dict) and "status" in evidence:
                details.append(f"evidence={evidence['status']}")
            if encoder.concept_adapter._public_spatial_prior is not None:
                details.append("public_spatial_prior")
            print(f"  [Pipeline] FDBR enabled ({', '.join(details)})")
        elif cfg.dsr_calibration_path:
            from .cipher_native.subspace_defense import DiagnosticSubspaceRetention
            dsr = DiagnosticSubspaceRetention.load(
                cfg.dsr_calibration_path,
                expected_sha256=checkpoint_hash(cfg.dsr_calibration_path),
            )
            encoder.concept_adapter.apply_dsr(dsr.projection, space="input")
            encoder.concept_adapter.prepare_cipher_weights()

        # Step 5: Build CKKS engines
        client_engine = ClientCANREngine(
            poly_modulus_degree=cfg.ckks_poly_modulus_degree,
            scale_bits=cfg.ckks_scale_bits,
            depth_budget=cfg.ckks_depth_budget,
            generate_relin_keys=not (
                DeploymentProfile(cfg.deployment_profile)
                is DeploymentProfile.MODEL_CONFIDENTIAL
                and cfg.poly_norm_degree == 1
                and max(cfg.poly_act_degree_schedule) == 1
            ),
        )
        server_engine = ServerCANREngine(client_engine)

        # Step 6: Load LLM (optional — None skips LLM for encoder-only experiments)
        if llm_name is not None:
            llm = LLMRegistry.load(
                llm_name,
                model_path=llm_model_path,
                allow_remote_code=cfg.allow_remote_model_code,
                remote_code_sha256=cfg.trusted_remote_code_sha256,
            )
        else:
            llm = None  # encoder-only mode (for FHE accuracy experiments)

        # Step 7: Load CFP sensitivity
        cfp_sens = None
        if cfp_sensitivity and os.path.exists(cfp_sensitivity):
            cfp_sens = load_tensor_checkpoint(
                cfp_sensitivity,
                expected_sha256=checkpoint_hash(cfp_sensitivity),
            )
            # Sensitivity files store the tensor under this explicit key.
            # "diagnostic_sensitivity" (plus metadata), not a bare tensor.
            # Unwrap before passing to DPCFPAccountant, which calls .float()
            # on it. Without this unwrap, pipeline build crashes with
            # "AttributeError: 'dict' object has no attribute 'float'".
            if isinstance(cfp_sens, dict) and "diagnostic_sensitivity" in cfp_sens:
                cfp_sens = cfp_sens["diagnostic_sensitivity"]

        pipeline = cls(
            encoder=encoder,
            llm=llm,
            client_engine=client_engine,
            server_engine=server_engine,
            medical_priors=priors,
            cfp_sensitivity=cfp_sens,
            config=cfg,
        )
        # Keep the registry name for trained-adapter lookup in run().
        pipeline._llm_name = llm_name
        pipeline._artifact_hashes = artifact_hashes
        probe_path = cfg.finding_probe_path
        if probe_path:
            from .cipher_native.finding_probe import FindingProbe

            pipeline.finding_probe = FindingProbe.load(
                probe_path,
                expected_sha256=checkpoint_hash(probe_path)
                if artifact_hashes
                else None,
            )
        return pipeline

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image to [1, 3, 224, 224] normalized tensor.

        Args:
            image_path: Path to image file (PNG/JPG).

        Returns:
            [1, 3, H, W] tensor on device, normalized to [-input_range, input_range].
        """
        return canonical_preprocess_image(
            image_path,
            image_size=self.config.image_size,
            input_range=self.config.input_range,
        ).to(self.config.device)

    def _prepare_prompt(self, prompt: Optional[str]) -> Optional[str]:
        """Add fixed anatomical anchors without exposing coefficient values."""
        if (
            not self.config.use_anatomical_prompt_anchors
            or self.config.release_slot_mode != "anatomical"
        ):
            return prompt
        anchors = ", ".join(
            slot.prompt_anchor
            for slot in canonical_anatomical_slots(
                grid_size=self.config.grid_size,
                include_background=self.config.anatomical_slot_include_background,
            )
        )
        prefix = f"Evaluate findings by region: {anchors}."
        return f"{prefix} {prompt}".strip() if prompt else prefix

    def run(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        return_metadata: bool = False,
        return_fhe_features: bool = False,
        use_cipher: bool = True,
    ) -> "PipelineResult":
        """End-to-end FHE-VLM inference: image → report.

        Args:
            image_path: Path to medical image.
            prompt: Optional text prompt (None = LLM default).
            return_metadata: If True, return full metadata (timings, DP report, etc.).
            return_fhe_features: If True, expose intermediate FHE features
                (fhe_16_tokens pre-DPCFP, fhe_256_tokens post-Perceiver pre-adapter)
                in PipelineResult. Used by experiments to cache FHE compute for
                reuse across multiple LLMs (avoids redundant FHE forwards).

        Returns:
            PipelineResult containing report text + metadata.
        """
        if (
            not use_cipher
            and self.deployment_profile is not DeploymentProfile.RESEARCH_BASELINE
        ):
            raise ValueError(
                "plaintext inference is restricted to research_baseline; "
                "deployment profiles require the encrypted vision path"
            )
        if return_fhe_features and self.deployment_profile is not DeploymentProfile.RESEARCH_BASELINE:
            raise ValueError(
                "return_fhe_features is a research debugging interface and is "
                "disabled for deployment profiles"
            )
        timings: Dict[str, float] = {}
        total_start = time.time()
        prompt = self._prepare_prompt(prompt)

        # ===== Step 1: Client preprocesses image =====
        t0 = time.time()
        image_tensor = self.preprocess_image(image_path)
        timings["client_preprocess"] = time.time() - t0

        # Patchify on client. In model-confidential mode the client does not
        # receive or apply the private patch projection or positional embedding.
        # FHE encoder runs on CPU (TenSEAL is CPU-only). Move image to CPU
        # for patchify + patch_proj (client-side plaintext ops). The LLM is
        # loaded separately on cuda via LLMRegistry (device_map="auto").
        t0 = time.time()
        with torch.no_grad():
            image_cpu = image_tensor.to("cpu")
            patches = self.encoder.patchify(image_cpu)  # [1, num_patches, patch_dim]
            x_flat = patches.reshape(self.config.num_patches, -1)
            if self.deployment_profile is DeploymentProfile.MODEL_CONFIDENTIAL:
                encrypted_input = x_flat
            else:
                encrypted_input = self.encoder.patch_proj(x_flat)
                encrypted_input = encrypted_input + self.encoder.pos_embed[
                    :, :self.config.num_patches, :
                ].squeeze(0)
        timings["client_patch_embed"] = time.time() - t0

        bottleneck = self.encoder.concept_adapter.diagnostic_bottleneck
        if not use_cipher:
            t0 = time.time()
            with torch.no_grad():
                encoder_device = next(self.encoder.parameters()).device
                vision_tokens = self.encoder.forward_vision_tokens(
                    image_tensor.to(encoder_device)
                )
                if bottleneck is not None:
                    fhe_16_tokens = bottleneck.encode_plain(vision_tokens).reshape(
                        -1, bottleneck.release_dim
                    )
                else:
                    fhe_16_tokens = self.encoder.concept_adapter.W_concept(
                        vision_tokens
                    ).reshape(-1, self.config.llm_hidden_dim)
            timings["client_encrypt"] = 0.0
            timings["server_fhe_forward"] = time.time() - t0
            timings["client_decrypt"] = 0.0
            self.encoder._last_refresh_stats = {
                "cadn_round_trips": 0,
                "cadn_token_refreshes": 0,
                "lbr_enabled": False,
                "pbsr_enabled": False,
                "mode": "plain_student",
            }
        else:
            # ===== Step 2: Client encrypts (real CKKS) =====
            t0 = time.time()
            raw_patch_layout = None
            if self.deployment_profile is DeploymentProfile.MODEL_CONFIDENTIAL:
                # Raw patches can exceed N/2 slots (e.g. 9408 features at
                # N=16384).  Encrypt chunked rows; the server recombines the
                # partial patch projections homomorphically.
                encrypted_tokens, raw_patch_layout = self.client_engine.encrypt_chunked(
                    encrypted_input
                )
            else:
                encrypted_tokens = self.client_engine.encrypt(encrypted_input)
            timings["client_encrypt"] = time.time() - t0

            ct_input = CiphertextTensor(
                ciphertexts=encrypted_tokens,
                engine=self.server_engine,
                depth=0,
                raw_patch_layout=raw_patch_layout,
            )

            # ===== Step 3: Server runs FHE ViT + CADN =====
            t0 = time.time()
            with torch.no_grad():
                if self.deployment_profile is DeploymentProfile.MODEL_CONFIDENTIAL:
                    ct_output = self.encoder.forward(
                        ct_input,
                        cspe_mode=False,
                        refresh_policy="none",
                        pre_cadn_noise_sigma=0,
                    )
                else:
                    from .cipher_native.pbsr_scheduler import PBSRScheduler
                    from .cipher_native.refresh_controller import RefreshController

                    pbsr_mask = None
                    if self.config.pbsr_enabled:
                        pbsr_mask = PBSRScheduler(
                            num_tokens=self.config.num_patches,
                            grid_size=int(self.config.num_patches ** 0.5),
                            coverage_target=self.config.pbsr_coverage,
                        ).compute_mask()
                    refresh_controller = RefreshController(
                        schedule=self.config._lbr_schedule,
                        mask=pbsr_mask,
                        num_tokens=self.config.num_patches,
                    )
                    ct_output = self.encoder.forward(
                        ct_input,
                        client_callback=self.client_engine.canr_refresh,
                        cspe_mode=True,
                        cadn_callback=self.client_engine.cadn_refresh,
                        pre_cadn_noise_sigma=self.config.pre_cadn_noise_sigma,
                        refresh_policy="interactive",
                        refresh_controller=refresh_controller,
                    )
            timings["server_fhe_forward"] = time.time() - t0

            # ===== Step 4: Client decrypts the configured release representation =====
            t0 = time.time()
            decrypted_dim = (
                bottleneck.release_dim
                if bottleneck is not None
                else self.config.llm_hidden_dim
            )
            fhe_16_tokens = self.client_engine.decrypt(
                ct_output._cipher, dim=decrypted_dim
            )
            timings["client_decrypt"] = time.time() - t0

        sigma_fhe = (
            self.noise_calibration.embedding_error_std
            if self.noise_calibration is not None
            else 0.0
        )

        # ===== Step 5: Client applies accounted DPCFP before plaintext release =====
        t0 = time.time()
        clean_release = fhe_16_tokens.unsqueeze(0)
        noised_release, dp_report = self.dp_accountant.apply_and_account(
            clean_release
        )
        governed_release = DiagnosticEmbeddingRelease(
            values=noised_release,
            dp_report=dp_report,
            deployment_profile=self.deployment_profile.value,
            calibrated=(
                self.deployment_profile is DeploymentProfile.RESEARCH_BASELINE
                or bottleneck is not None
            ),
        )
        timings["client_dpcfp"] = time.time() - t0

        # ===== Step 6: Server Perceiver decode (plaintext, private weights) =====
        t0 = time.time()
        with torch.no_grad():
            if bottleneck is not None:
                noised_embedding = self.encoder.concept_adapter.server_side_bottleneck_decode(
                    governed_release.values
                )
            else:
                noised_embedding = governed_release.values
            fhe_256_tokens = self.encoder.concept_adapter.server_side_perceiver_decode(
                noised_embedding
            )
        timings["server_perceiver"] = time.time() - t0

        # Cache intermediate FHE features for experiment reuse (Phase C: same
        # FHE features feed all 4 LLMs, avoiding 4× redundant FHE forwards).
        # fhe_16_cached: pre-CFP (clean FHE, for cos vs plain)
        # fhe_256_cached: post-CFP (noised, what LLM actually sees)
        # fhe_256_clean_cached: pre-CFP perceiver output (clean FHE, for cos vs plain)
        fhe_16_cached = fhe_16_tokens.detach().cpu() if return_fhe_features else None
        fhe_256_cached = fhe_256_tokens.detach().cpu() if return_fhe_features else None
        fhe_256_clean_cached = None
        if return_fhe_features:
            with torch.no_grad():
                # Only the explicitly requested research debugging path may
                # pass an unsanitized release to server-side decoder code.
                clean_embedding = (
                    self.encoder.concept_adapter.server_side_bottleneck_decode(clean_release)
                    if bottleneck is not None else clean_release
                )
                fhe_256_clean_cached = self.encoder.concept_adapter.server_side_perceiver_decode(
                    clean_embedding
                ).detach().cpu()

        # Encoder-only mode skips LLM steps when no LLM is loaded.
        # Used by FHE accuracy experiments (cos measurement, defense impact)
        # that don't need text generation. Avoids loading 4B-param LLM.
        if self.llm is None:
            timings["total"] = time.time() - total_start
            canr_stats = self.client_engine.get_stats()
            return PipelineResult(
                report="",
                dp_report=dp_report,
                timings=timings,
                canr_refresh_count=canr_stats.get("refresh_count", 0),
                fhe_depth=(
                    self.encoder.continuous_multiplicative_depth()
                    if self.deployment_profile is DeploymentProfile.MODEL_CONFIDENTIAL
                    else self.encoder.multiplicative_depth()
                ),
                generation_metadata={
                    "mode": "encoder_only",
                    "sigma_fhe_calibrated": sigma_fhe,
                },
                embedding_metadata={
                    "release_token_count": governed_release.token_count,
                    "release_dim": governed_release.release_dim,
                    "release_values_per_image": governed_release.values_per_image,
                    "release_slot_mode": self.config.release_slot_mode,
                    "anatomical_prompt_anchors": bool(
                        self.config.release_slot_mode == "anatomical"
                        and self.config.use_anatomical_prompt_anchors
                    ),
                    "anatomical_slot_names": (
                        [slot.name for slot in canonical_anatomical_slots(
                            grid_size=self.config.grid_size,
                            include_background=self.config.anatomical_slot_include_background,
                        )]
                        if self.config.release_slot_mode == "anatomical"
                        else None
                    ),
                    "llm_token_count": fhe_256_tokens.shape[1],
                    "llm_hidden_dim": fhe_256_tokens.shape[2],
                    "llm_name": None,
                    "refresh_stats": getattr(self.encoder, "_last_refresh_stats", {}),
                },
                fhe_16_tokens=fhe_16_cached,
                fhe_256_tokens=fhe_256_cached,
                fhe_256_clean_tokens=fhe_256_clean_cached,
                lbr_schedule_summary=(
                    self.config._lbr_schedule.summary()
                    if self.config._lbr_schedule is not None
                    else None
                ),
            )

        # ===== Step 6b: load a trained adapter or explicit zero-shot ablation =====
        # When the LLM's hidden_dim != encoder output dim (2560), apply cross-LLM
        # adaptation. v174 prioritizes TRAINED adapters (distilled from each LLM's
        # native vision tower) over zero-shot PCA. Trained adapters preserve the
        # semantic structure needed by each LLM's vision tower subspace.
        #
        # A trained adapter is required unless zero-shot PCA is explicitly
        # enabled as an ablation.
        t0 = time.time()
        if self.config.use_llm_embedding_adapter:
            llm_dim = self.llm.llm_hidden_dim
            encoder_dim = fhe_256_tokens.shape[-1]
            if llm_dim != encoder_dim:
                from fpavlm.llm.embedding_adapter import LLMEmbeddingAdapter
                # Use cached adapter if available (avoids re-computing every call)
                if not hasattr(self, '_llm_adapter_cache'):
                    self._llm_adapter_cache = {}
                cache_key = f"{encoder_dim}->{llm_dim}"
                if cache_key not in self._llm_adapter_cache:
                    # Check for a trained adapter distilled from the native vision tower.
                    # Installed users should configure adapters_dir explicitly.
                    llm_name = getattr(self, '_llm_name', '')
                    trained_adapter_path = None
                    if llm_name:
                        if self.config.adapters_dir is not None:
                            adapters_base = self.config.adapters_dir
                        else:
                            adapters_base = os.path.join(
                                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "artifacts",
                                "adapters",
                            )
                        names = [f"{llm_name}.pt"]
                        if bottleneck is None:
                            names.insert(0, f"{llm_name}-fullstudent.pt")
                        for name in names:
                            candidate = os.path.join(adapters_base, name)
                            if os.path.exists(candidate):
                                trained_adapter_path = candidate
                                break

                    if trained_adapter_path is not None:
                        # Load the trained adapter. A corrupt artifact must fail
                        # closed: raise immediately rather than silently falling
                        # back to zero-shot PCA (which is only an explicit ablation).
                        try:
                            adapter_key = os.path.normcase(
                                os.path.abspath(trained_adapter_path)
                            )
                            expected_hash = getattr(self, "_artifact_hashes", {}).get(
                                adapter_key
                            )
                            if getattr(self, "_artifact_hashes", {}) and expected_hash is None:
                                raise ValueError(
                                    f"Adapter {trained_adapter_path!r} is absent from "
                                    "the configured artifact manifest"
                                )
                            adapter = LLMEmbeddingAdapter.load(
                                trained_adapter_path,
                                expected_sha256=expected_hash,
                            )
                            adapter.validate_compatibility(
                                target_llm_name=llm_name,
                                source_dim=self.config.llm_hidden_dim,
                                target_dim=self.llm.llm_hidden_dim,
                                num_tokens=self.config.target_grid ** 2,
                                require_provenance=(
                                    self.deployment_profile
                                    is not DeploymentProfile.RESEARCH_BASELINE
                                ),
                            )
                            adapter = adapter.to(self.config.device)
                            adapter.eval()
                        except (OSError, RuntimeError, ValueError, KeyError) as exc:
                            raise RuntimeError(
                                f"Failed to load required trained adapter {trained_adapter_path}"
                            ) from exc

                    if trained_adapter_path is None:
                        if not self.config.allow_zero_shot_adapter:
                            raise FileNotFoundError(
                                f"No trained adapter found for {llm_name!r}; expected "
                                f"{llm_name}.pt in {adapters_base}. "
                                "Set allow_zero_shot_adapter=True only for an explicit ablation."
                            )
                        # Explicit ablation: zero-shot PCA + stats matching.
                        adapter = LLMEmbeddingAdapter(
                            source_dim=encoder_dim,
                            target_dim=llm_dim,
                            num_tokens=fhe_256_tokens.shape[1],
                        )
                        adapter = adapter.to(self.config.device)
                        adapter.eval()
                        adapter.init_pca_from_features(fhe_256_tokens.to(self.config.device))
                        native_features = self.llm.get_native_feature_stats()
                        if native_features is not None:
                            adapter.match_target_statistics(
                                fhe_256_tokens.to(self.config.device), native_features
                            )
                    self._llm_adapter_cache[cache_key] = adapter
                adapter = self._llm_adapter_cache[cache_key]
                with torch.no_grad():
                    fhe_256_tokens = adapter(fhe_256_tokens.to(self.config.device))
        timings["server_llm_adapter"] = time.time() - t0

        visual_embedding = LLMVisualEmbedding(
            values=fhe_256_tokens,
            source_release_dim=governed_release.release_dim,
            llm_name=getattr(self, "_llm_name", self.llm.__class__.__name__),
        )
        visual_embedding.validate_for(self.llm)
        fhe_256_tokens = visual_embedding.values

        finding_hint = None
        if self.finding_probe is not None and bottleneck is not None:
            finding_hint = self.finding_probe.describe(governed_release.values)
            if finding_hint:
                prompt = f"{finding_hint} {prompt}".strip() if prompt else finding_hint

        # ===== Step 7: Server LLM generation (with CNAD noise-aware decoding) =====
        t0 = time.time()
        # CNAD combines measured-noise temperature, top-sigma, and MBR controls.
        # CNAD is opt-in and requires a paired noise calibration.
        # KNOWN, DETERMINISTIC, PER-TOKEN structure of FHE noise — bridging FHE
        # cryptography with LLM decoding. It falls back to generic noise-aware generation
        # (generic NLP decoding) when use_cnad_decoding=False, and to plain beam
        # search when use_mbr_decoding=False.
        # sigma_fhe is measured by the configured calibration, not hardcoded.
        if self.config.use_cnad_decoding:
            report_text, gen_meta = cnad_generate(
                self.llm,
                fhe_256_tokens,
                prompt=prompt,
                max_new_tokens=self.config.llm_max_new_tokens,
                sigma_fhe_embedding=sigma_fhe,
                k=self.config.mbr_k,
                beam_width=max(self.config.mbr_beam_width, self.config.llm_num_beams),
                seed=self.config.llm_seed,
            )
        elif self.config.use_mbr_decoding:
            report_text, gen_meta = noise_aware_generate(
                self.llm,
                fhe_256_tokens,
                prompt=prompt,
                max_new_tokens=self.config.llm_max_new_tokens,
                use_mbr=True,
                k=self.config.mbr_k,
                beam_width=self.config.mbr_beam_width,
                sigma_fhe_embedding=sigma_fhe,
                seed=self.config.llm_seed,
            )
        else:
            report_text = self.llm.generate(
                fhe_256_tokens, prompt=prompt,
                max_new_tokens=self.config.llm_max_new_tokens,
                do_sample=False, num_beams=self.config.llm_num_beams,
                seed=self.config.llm_seed,
            )
            gen_meta = {"mechanisms": ["plain_beam"], "mbr_report": None,
                       "sigma_fhe_calibrated": sigma_fhe}
        if finding_hint:
            gen_meta["finding_hint"] = finding_hint
        timings["server_llm_generate"] = time.time() - t0

        timings["total"] = time.time() - total_start

        canr_stats = self.client_engine.get_stats()

        result = PipelineResult(
            report=report_text,
            dp_report=dp_report,
            timings=timings,
            canr_refresh_count=canr_stats.get("refresh_count", 0),
            fhe_depth=(
                self.encoder.continuous_multiplicative_depth()
                if self.deployment_profile is DeploymentProfile.MODEL_CONFIDENTIAL
                else self.encoder.multiplicative_depth()
            ),
            generation_metadata=gen_meta,
            embedding_metadata={
                "release_token_count": governed_release.token_count,
                "release_dim": governed_release.release_dim,
                "release_values_per_image": governed_release.values_per_image,
                "release_slot_mode": self.config.release_slot_mode,
                "anatomical_prompt_anchors": bool(
                    self.config.release_slot_mode == "anatomical"
                    and self.config.use_anatomical_prompt_anchors
                ),
                "anatomical_slot_names": (
                    [slot.name for slot in canonical_anatomical_slots(
                        grid_size=self.config.grid_size,
                        include_background=self.config.anatomical_slot_include_background,
                    )]
                    if self.config.release_slot_mode == "anatomical"
                    else None
                ),
                "llm_token_count": visual_embedding.values.shape[1],
                "llm_hidden_dim": visual_embedding.values.shape[2],
                "llm_name": visual_embedding.llm_name,
                "refresh_stats": getattr(self.encoder, "_last_refresh_stats", {}),
            },
            fhe_16_tokens=fhe_16_cached,
            fhe_256_tokens=fhe_256_cached,
            fhe_256_clean_tokens=fhe_256_clean_cached,
            llm_input_tokens=(
                visual_embedding.values.detach().cpu() if return_fhe_features else None
            ),
            lbr_schedule_summary=(
                self.config._lbr_schedule.summary()
                if self.config._lbr_schedule is not None
                else None
            ),
        )
        return result


@dataclass
class PipelineResult:
    """Result of FHEViTPipeline.run() — report + full metadata."""

    report: str
    dp_report: DPReport
    timings: Dict[str, float]
    canr_refresh_count: int
    fhe_depth: int
    generation_metadata: Dict[str, Any]
    embedding_metadata: Dict[str, Any] = field(default_factory=dict)
    # Intermediate FHE features (only populated when return_fhe_features=True).
    # fhe_16_tokens: [16, llm_hidden_dim] — 16-token embedding pre-DPCFP (privacy analysis)
    # Noised post-Perceiver embedding, before any LLM-specific adapter.
    # fhe_256_clean_tokens: [1, 256, encoder_dim] — 256-token embedding post-Perceiver pre-CFP (clean, for cos vs plain)
    fhe_16_tokens: Optional[torch.Tensor] = None
    fhe_256_tokens: Optional[torch.Tensor] = None
    fhe_256_clean_tokens: Optional[torch.Tensor] = None
    llm_input_tokens: Optional[torch.Tensor] = None
    # LBR/PBSR optimization metadata (populated when lbr_enabled=True)
    lbr_schedule_summary: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"PipelineResult(report_len={len(self.report)}, "
            f"dp={self.dp_report}, "
            f"total_time={self.timings.get('total', 0):.1f}s, "
            f"fhe_depth={self.fhe_depth}, "
            f"canr_refreshes={self.canr_refresh_count})"
        )

    def composed_epsilon(self, releases: int) -> float:
        """Conservative basic-composition epsilon for repeated calls."""
        return self.dp_report.composed_epsilon(releases)

    def summary(self) -> str:
        """Human-readable summary for display."""
        lines = [
            "=" * 70,
            "FHEViTPipeline Result Summary",
            "=" * 70,
            f"Report: {self.report[:200]}{'...' if len(self.report) > 200 else ''}",
            "",
            "Privacy Guarantee: "
            + (
                self.dp_report.guarantee_str
                if hasattr(self.dp_report, "guarantee_str")
                else str(self.dp_report)
            ),
            f"FHE Depth: {self.fhe_depth}",
            f"CANR Refreshes: {self.canr_refresh_count}",
            "",
            "Timings (seconds):",
        ]
        for k, v in self.timings.items():
            lines.append(f"  {k:30s}: {v:8.2f}")
        lines.append("=" * 70)
        return "\n".join(lines)


__all__ = [
    "FHEConfig",
    "FHEViTPipeline",
    "PipelineResult",
    "cli_quickstart",
]


# =============================================================================
# CLI Entry Point (installed as `fpavlm-quickstart` via [project.scripts])
# =============================================================================

def cli_quickstart():
    """CLI entry point for `fpavlm-quickstart` command.

    Usage:
        fpavlm-quickstart --image path/to/xray.png
        fpavlm-quickstart  # uses default OpenI sample
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="fpavlm-quickstart",
        description="FHEViT one-line API: privacy-preserving medical VLM inference",
    )
    parser.add_argument("--image", default=None, help="Path to chest X-ray image")
    parser.add_argument("--llm", default="medgemma-4b", help="LLM registry name")
    parser.add_argument("--llm-path", default="models/unsloth/medgemma-4b-it", help="LLM weights path")
    parser.add_argument("--encoder-weights", default="artifacts/encoder_weights.pt")
    parser.add_argument("--medical-priors", default="artifacts/medical_priors.pt")
    parser.add_argument("--cfp-sensitivity", default="artifacts/cfp_sensitivity.pt")
    parser.add_argument("--cnad", action="store_true", help="Enable calibrated CNAD")
    parser.add_argument("--mbr", action="store_true", help="Enable calibrated MBR")
    parser.add_argument(
        "--noise-calibration",
        default=None,
        help="Paired FHE/plain release calibration JSON required by CNAD or MBR",
    )
    parser.add_argument("--no-quantize", action="store_true", help="Disable int8 quantization")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    # Default image
    if args.image is None:
        args.image = os.path.join("data", "openi_subset", "images", "openi_00001.png")

    config = FHEConfig(
        use_cadn=True,
        cfp_noise_fraction=24.0,
        cfp_clip_norm=1.0,
        cfp_delta=1e-5,
        cfp_quantize=not args.no_quantize,
        use_cnad_decoding=args.cnad,
        use_mbr_decoding=args.mbr,
        noise_calibration_path=args.noise_calibration,
        mbr_k=4,
        mbr_beam_width=2,
        llm_num_beams=4,
        llm_seed=42,
    )
    if args.cpu:
        config.device = "cpu"

    print("=" * 70)
    print("FHEViT Quickstart — Privacy-Preserving Medical VLM")
    print("=" * 70)
    print(f"Image: {args.image}")
    print(f"LLM: {args.llm}")
    print(f"FHE: CKKS N={config.ckks_poly_modulus_degree}, scale=2^{config.ckks_scale_bits}")
    print(f"CADN: {config.use_cadn}")
    print(f"CFP: σ={config.cfp_noise_fraction}×, clip R={config.cfp_clip_norm}, δ={config.cfp_delta}")
    print(f"Quantize int8: {config.cfp_quantize}")
    print(f"MBR decoding: {config.use_mbr_decoding} (k={config.mbr_k}, beam={config.mbr_beam_width})")
    print("=" * 70)

    # Build pipeline (one-time setup, ~30s)
    print("\n[1/2] Loading pipeline (encoder + LLM + FHE context)...")
    t0 = time.time()
    pipeline = FHEViTPipeline.from_pretrained(
        llm_name=args.llm,
        llm_model_path=args.llm_path,
        encoder_weights=args.encoder_weights,
        medical_priors=args.medical_priors,
        cfp_sensitivity=args.cfp_sensitivity,
        config=config,
    )
    print(f"    Loaded in {time.time() - t0:.1f}s")

    # Run inference (one-line API!)
    print(f"\n[2/2] Running FHE-VLM inference on {os.path.basename(args.image)}...")
    t0 = time.time()
    result = pipeline.run(args.image)
    elapsed = time.time() - t0

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nReport:\n{result.report}")
    print(f"\nPrivacy Guarantee: {result.dp_report}")
    print(f"FHE Depth: {result.fhe_depth}")
    print(f"CANR Refreshes: {result.canr_refresh_count}")
    print("\nTimings:")
    for k, v in result.timings.items():
        print(f"  {k:30s}: {v:8.2f}s")
    print(f"\nTotal inference time: {elapsed:.1f}s")
    print("=" * 70)
