"""Public cipher-native components used by FHEViT."""

from .adaptive_noise import AdaptiveCKKSNoise
from .anatomy_router import AnatomyAwareTokenRouter
from .base_perceiver import BasePerceiverDecoder, PerceiverRegistry
from .cadn_norm import CADNLayer
from .canr import ClientCANREngine, ServerCANREngine
from .cipher_norm import CipherRMSNorm
from .ciphertext_tensor import CiphertextTensor
from .demographic_filter import (
    DemographicNullSpaceFilter,
    LesionAwareDemNull,
    apply_demographic_null_space,
    apply_lesion_aware_demographic_null_space,
    compute_demographic_subspace,
    compute_lesion_aware_demographic_subspace,
    estimate_expected_attribute_auc_reduction,
    filter_release_residual,
)
from .diagnostic_bottleneck import BottleneckMetadata, DiagnosticBottleneckRelease
from .dp_accountant import (
    DPCFPAccountant,
    DPReport,
    compute_dp_guarantee,
    l2_clip,
    quantize_int8,
    sigma_for_target_epsilon,
)
from .finding_probe import FindingProbe, fit_ridge
from .identity_perceiver import IdentityPerceiver
from .lbr_scheduler import (
    BlockRefreshDecision,
    LazyBlockRefreshScheduler,
    LBRSchedule,
    default_lbr_schedule,
)
from .mafo import mafo_select_optimal_scale
from .med_fhe_vit_block import MedFHEBlock
from .medical_concept_adapter import MedicalConceptAdapter
from .noise_resilient_encoder import NoiseResilientMAViTEncoder
from .pbsr_scheduler import (
    PBSRMask,
    PBSRScheduler,
    compute_pbsr_lbr_speedup,
)
from .perceiver_decoder import (
    CrossAttentionLayer,
    MagnitudeCalibratedNorm,
    PerceiverDecoder,
)
from .poly_norm import PolyNorm
from .refresh_controller import (
    RefreshAction,
    RefreshController,
    merge_ciphertexts,
    select_ciphertexts,
)
from .spectral_bounded import SpectralBoundedLinear

__all__ = [
    "AdaptiveCKKSNoise",
    "AnatomyAwareTokenRouter",
    "BasePerceiverDecoder",
    "BlockRefreshDecision",
    "BottleneckMetadata",
    "CADNLayer",
    "CipherRMSNorm",
    "CiphertextTensor",
    "ClientCANREngine",
    "CrossAttentionLayer",
    "DPCFPAccountant",
    "DPReport",
    "DemographicNullSpaceFilter",
    "LesionAwareDemNull",
    "DiagnosticBottleneckRelease",
    "FindingProbe",
    "IdentityPerceiver",
    "LBRSchedule",
    "LazyBlockRefreshScheduler",
    "MagnitudeCalibratedNorm",
    "MedFHEBlock",
    "MedicalConceptAdapter",
    "NoiseResilientMAViTEncoder",
    "PBSRMask",
    "PBSRScheduler",
    "RefreshAction",
    "RefreshController",
    "PerceiverDecoder",
    "PerceiverRegistry",
    "PolyNorm",
    "ServerCANREngine",
    "SpectralBoundedLinear",
    "apply_demographic_null_space",
    "apply_lesion_aware_demographic_null_space",
    "compute_demographic_subspace",
    "compute_lesion_aware_demographic_subspace",
    "compute_dp_guarantee",
    "compute_pbsr_lbr_speedup",
    "default_lbr_schedule",
    "merge_ciphertexts",
    "select_ciphertexts",
    "estimate_expected_attribute_auc_reduction",
    "filter_release_residual",
    "fit_ridge",
    "l2_clip",
    "mafo_select_optimal_scale",
    "quantize_int8",
    "sigma_for_target_epsilon",
]
