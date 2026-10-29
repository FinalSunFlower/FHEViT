"""Public API for the FHEViT medical FHE-VLM research framework.

The default configuration is a research baseline. Deployment profiles enforce
explicit release calibration, target epsilon, and artifact integrity.
"""

from fpavlm.affine_release import CompiledAffineRelease, compile_affine_release
from fpavlm.anatomical_slots import (
    CANONICAL_CXR_SLOT_NAMES,
    AnatomicalSlot,
    canonical_anatomical_slots,
    slot_token_groups,
    validate_slot_partition,
)
from fpavlm.cipher_native import (
    AdaptiveCKKSNoise,
    AnatomyAwareTokenRouter,
    BasePerceiverDecoder,
    BottleneckMetadata,
    CADNLayer,
    CipherRMSNorm,
    CiphertextTensor,
    ClientCANREngine,
    CrossAttentionLayer,
    DiagnosticBottleneckRelease,
    IdentityPerceiver,
    LesionAwareDemNull,
    MagnitudeCalibratedNorm,
    MedFHEBlock,
    MedicalConceptAdapter,
    NoiseResilientMAViTEncoder,
    PerceiverDecoder,
    PerceiverRegistry,
    PolyNorm,
    ServerCANREngine,
    SpectralBoundedLinear,
    apply_lesion_aware_demographic_null_space,
    compute_lesion_aware_demographic_subspace,
    mafo_select_optimal_scale,
)
from fpavlm.cipher_native.dp_accountant import (
    DPCFPAccountant,
    DPReport,
    compute_dp_guarantee,
    l2_clip,
    quantize_int8,
    sigma_for_target_epsilon,
)
from fpavlm.deployment import DeploymentProfile
from fpavlm.diagnostic_prefix import DiagnosticPrefixProjector, PrefixSpec
from fpavlm.embedding_contract import DiagnosticEmbeddingRelease, LLMVisualEmbedding
from fpavlm.llm import (
    InternVL2Adapter,
    LLMEmbeddingAdapter,
    LLMInterface,
    LLMRegistry,
    Qwen2VLAdapter,
)
from fpavlm.llm.noise_aware_decoding import (
    TopNSigmaLogitProcessor,
    UncertaintyBoundedLogitProcessor,
    noise_aware_generate,
)
from fpavlm.pipeline import FHEConfig, FHEViTPipeline, PipelineResult
from fpavlm.privacy_audit import (
    PatientLevelPrivacyAuditor,
    PrivacyAuditThresholds,
    RepresentationPrivacyAudit,
    bind_privacy_audit_to_calibration,
    validate_privacy_audit,
    validate_release_calibration,
)
from fpavlm.spatial_quadratic import CompiledSpatialQuadratic, SpatialQuadraticRelease
from fpavlm.vision_plugin import FHEVisionCapabilities, FHEVisionRegistry

__version__ = "0.4.0"

__all__ = [
    "DiagnosticPrefixProjector",
    "PrefixSpec",
    "CompiledAffineRelease",
    "CompiledSpatialQuadratic",
    "SpatialQuadraticRelease",
    "compile_affine_release",
    "AdaptiveCKKSNoise",
    "AnatomicalSlot",
    "AnatomyAwareTokenRouter",
    "BasePerceiverDecoder",
    "BottleneckMetadata",
    "CADNLayer",
    "CANONICAL_CXR_SLOT_NAMES",
    "CipherRMSNorm",
    "CiphertextTensor",
    "ClientCANREngine",
    "CrossAttentionLayer",
    "DPCFPAccountant",
    "DPReport",
    "DeploymentProfile",
    "DiagnosticEmbeddingRelease",
    "DiagnosticBottleneckRelease",
    "FHEConfig",
    "FHEViTPipeline",
    "FHEVisionCapabilities",
    "FHEVisionRegistry",
    "canonical_anatomical_slots",
    "IdentityPerceiver",
    "InternVL2Adapter",
    "LesionAwareDemNull",
    "LLMEmbeddingAdapter",
    "LLMInterface",
    "LLMRegistry",
    "LLMVisualEmbedding",
    "MagnitudeCalibratedNorm",
    "MedFHEBlock",
    "MedicalConceptAdapter",
    "NoiseResilientMAViTEncoder",
    "PerceiverDecoder",
    "PerceiverRegistry",
    "PatientLevelPrivacyAuditor",
    "PipelineResult",
    "PolyNorm",
    "Qwen2VLAdapter",
    "PrivacyAuditThresholds",
    "RepresentationPrivacyAudit",
    "ServerCANREngine",
    "SpectralBoundedLinear",
    "TopNSigmaLogitProcessor",
    "UncertaintyBoundedLogitProcessor",
    "bind_privacy_audit_to_calibration",
    "apply_lesion_aware_demographic_null_space",
    "compute_lesion_aware_demographic_subspace",
    "compute_dp_guarantee",
    "l2_clip",
    "mafo_select_optimal_scale",
    "noise_aware_generate",
    "quantize_int8",
    "sigma_for_target_epsilon",
    "slot_token_groups",
    "validate_privacy_audit",
    "validate_release_calibration",
    "validate_slot_partition",
]
