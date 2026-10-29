"""Capability-checked plug-in boundary for encrypted vision encoders."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Callable, Dict, Optional, Tuple

from fpavlm.deployment import DeploymentProfile

_ALLOWED_OPERATIONS = frozenset({
    "add",
    "cipher_plain_mul",
    "cipher_cipher_mul",
})


@dataclass(frozen=True)
class FHEVisionCapabilities:
    """Auditable properties a vision plug-in must declare before registration."""

    input_domain: str
    output_domain: str
    homomorphic_operations: Tuple[str, ...]
    supports_interactive: bool
    supports_callback_free: bool
    protects_server_weights_callback_free: bool
    supports_raw_patch_ciphertext: bool

    def __post_init__(self) -> None:
        if self.input_domain not in {"raw_patches", "projected_patches", "both"}:
            raise ValueError("unsupported FHE vision input domain")
        if self.output_domain != "diagnostic_coefficients":
            raise ValueError(
                "FHE vision plug-ins must emit encrypted diagnostic coefficients"
            )
        operations = set(self.homomorphic_operations)
        if not operations or operations - _ALLOWED_OPERATIONS:
            raise ValueError(
                "homomorphic_operations must be a non-empty subset of "
                f"{sorted(_ALLOWED_OPERATIONS)}"
            )
        if self.protects_server_weights_callback_free and not self.supports_callback_free:
            raise ValueError(
                "server-weight protection requires a callback-free execution path"
            )
        if self.supports_callback_free and not self.supports_raw_patch_ciphertext:
            raise ValueError(
                "callback-free execution must accept encrypted raw patch vectors"
            )

    def validate_profile(self, profile: DeploymentProfile) -> None:
        if profile is DeploymentProfile.UTILITY_INTERACTIVE and not self.supports_interactive:
            raise ValueError("vision plug-in does not support utility_interactive")
        if profile is DeploymentProfile.MODEL_CONFIDENTIAL:
            if not self.supports_callback_free:
                raise ValueError("vision plug-in does not support callback-free inference")
            if not self.protects_server_weights_callback_free:
                raise ValueError(
                    "vision plug-in does not protect server weights in callback-free mode"
                )
            if not self.supports_raw_patch_ciphertext:
                raise ValueError(
                    "model_confidential requires encrypted raw-patch input"
                )


@dataclass(frozen=True)
class RegisteredFHEVisionPlugin:
    name: str
    factory: Callable
    capabilities: FHEVisionCapabilities


class FHEVisionRegistry:
    """Registry that validates both declared capabilities and runtime methods."""

    _plugins: Dict[str, RegisteredFHEVisionPlugin] = {}
    _entry_points_loaded: bool = False
    _entry_points_seen: set[str] = set()

    @classmethod
    def discover(cls, *, force: bool = False) -> Tuple[str, ...]:
        """Load third-party FHE vision plug-ins from Python entry points.

        Packages expose an entry point in the ``fpavlm.fhe_vision`` group.  The
        entry point may be a module (registration happens on import) or a
        zero-argument registration function.  Discovery is explicit and
        idempotent so importing FHEViT never executes untrusted package code.
        """
        if cls._entry_points_loaded and not force:
            return cls.available()
        try:
            entries = metadata.entry_points()
            if hasattr(entries, "select"):
                entries = entries.select(group="fpavlm.fhe_vision")
            else:  # Python 3.9 compatibility
                entries = entries.get("fpavlm.fhe_vision", ())
        except Exception as exc:  # pragma: no cover - platform metadata edge case
            raise RuntimeError("could not enumerate FHE vision plug-in entry points") from exc
        before = set(cls._plugins)
        for entry in entries:
            entry_key = f"{getattr(entry, 'name', '')}:{getattr(entry, 'value', entry)}"
            if entry_key in cls._entry_points_seen:
                continue
            loaded = entry.load()
            if callable(loaded) and not isinstance(loaded, type):
                loaded()
            cls._entry_points_seen.add(entry_key)
        cls._entry_points_loaded = True
        return tuple(sorted(set(cls._plugins) - before))

    @classmethod
    def register(
        cls,
        name: str,
        *,
        capabilities: FHEVisionCapabilities,
    ) -> Callable[[Callable], Callable]:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("vision plug-in name must not be empty")

        def decorator(factory: Callable) -> Callable:
            if normalized in cls._plugins:
                raise ValueError(f"FHE vision plug-in {normalized!r} is already registered")
            cls._plugins[normalized] = RegisteredFHEVisionPlugin(
                normalized, factory, capabilities
            )
            return factory

        return decorator

    @classmethod
    def load(
        cls,
        name: str,
        *,
        deployment_profile: str,
        **kwargs,
    ):
        cls.discover()
        normalized = name.strip().lower()
        if normalized not in cls._plugins:
            raise KeyError(
                f"Unknown FHE vision plug-in {name!r}; available: {sorted(cls._plugins)}"
            )
        registration = cls._plugins[normalized]
        profile = DeploymentProfile(deployment_profile)
        registration.capabilities.validate_profile(profile)
        instance = registration.factory(**kwargs)
        cls.validate_instance(instance, profile=profile)
        return instance

    @classmethod
    def validate_instance(
        cls,
        instance,
        *,
        profile: DeploymentProfile,
        expected_continuous_depth: Optional[int] = None,
    ) -> None:
        required = (
            "patchify",
            "forward",
            "bake_statistics",
            "multiplicative_depth",
            "continuous_multiplicative_depth",
        )
        missing = [name for name in required if not callable(getattr(instance, name, None))]
        if missing:
            raise TypeError(f"FHE vision plug-in is missing methods: {missing}")
        if getattr(instance, "concept_adapter", None) is None:
            raise TypeError("FHE vision plug-in must expose concept_adapter")
        if profile is DeploymentProfile.MODEL_CONFIDENTIAL:
            depth = int(instance.continuous_multiplicative_depth())
            if depth < 1:
                raise ValueError("callback-free FHE depth must be positive")
            if expected_continuous_depth is not None and depth != expected_continuous_depth:
                raise ValueError(
                    f"vision plug-in reports callback-free depth {depth}, "
                    f"expected {expected_continuous_depth}"
                )

    @classmethod
    def capabilities(cls, name: str) -> FHEVisionCapabilities:
        cls.discover()
        normalized = name.strip().lower()
        if normalized not in cls._plugins:
            raise KeyError(f"Unknown FHE vision plug-in {name!r}")
        return cls._plugins[normalized].capabilities

    @classmethod
    def available(cls) -> Tuple[str, ...]:
        return tuple(sorted(cls._plugins))


__all__ = [
    "FHEVisionCapabilities",
    "FHEVisionRegistry",
    "RegisteredFHEVisionPlugin",
]
