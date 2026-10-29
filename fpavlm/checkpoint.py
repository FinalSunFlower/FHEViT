"""Fail-closed loading helpers for trusted local tensor checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

import torch


def load_tensor_checkpoint(
    path: str,
    *,
    required_keys: Optional[Iterable[str]] = None,
    expected_sha256: Optional[str] = None,
) -> Any:
    """Load a local tensor-only checkpoint and validate its top-level schema.

    `weights_only=True` rejects arbitrary pickle globals. Model directories that
    intentionally execute Transformers remote code are a separate trust decision.
    """
    resolved = os.path.abspath(os.fspath(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    if expected_sha256 is not None:
        expected = expected_sha256.lower()
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
        digest = hashlib.sha256()
        with open(resolved, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(
                f"Checkpoint digest mismatch for {resolved}: expected {expected}, got {actual}"
            )
    try:
        value = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This fpavlm release requires a torch version supporting "
            "torch.load(..., weights_only=True)"
        ) from exc
    if required_keys is not None:
        if not isinstance(value, Mapping):
            raise ValueError(f"Checkpoint {resolved} must contain a top-level mapping")
        missing = sorted(set(required_keys) - set(value))
        if missing:
            raise ValueError(f"Checkpoint {resolved} is missing required keys: {missing}")
    return value


def load_artifact_manifest(path: str) -> dict[str, str]:
    """Load a manifest as absolute-path to SHA-256 mappings.

    Artifact paths are resolved relative to the manifest, so verification does
    not depend on the process working directory.
    """
    resolved = os.path.abspath(os.fspath(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Artifact manifest not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("artifacts"), dict):
        raise ValueError("Artifact manifest must contain an 'artifacts' object")
    hashes = {}
    for artifact, digest in value["artifacts"].items():
        normalized = str(digest).lower()
        if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
            raise ValueError(f"Invalid SHA-256 digest for artifact {artifact!r}")
        artifact_path = os.fspath(artifact)
        if not os.path.isabs(artifact_path):
            artifact_path = os.path.join(os.path.dirname(resolved), artifact_path)
        canonical = os.path.normcase(os.path.abspath(artifact_path))
        if canonical in hashes:
            raise ValueError(f"Duplicate artifact path in manifest: {artifact!r}")
        hashes[canonical] = normalized
    return hashes


def promote_artifact(manifest_path: str, artifact_path: str) -> str:
    """Atomically bind a local artifact's current digest into its manifest."""
    manifest = Path(manifest_path).resolve()
    artifact = Path(artifact_path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Artifact manifest not found: {manifest}")
    if not artifact.is_file():
        raise FileNotFoundError(f"Artifact not found: {artifact}")
    try:
        relative = artifact.relative_to(manifest.parent).as_posix()
    except ValueError as exc:
        raise ValueError("artifact must be inside the manifest directory") from exc
    with manifest.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("hash_algorithm") != "sha256"
        or not isinstance(value.get("artifacts"), dict)
    ):
        raise ValueError("unsupported artifact manifest schema")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    value["artifacts"][relative] = digest
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest)
    return digest


__all__ = ["load_artifact_manifest", "load_tensor_checkpoint", "promote_artifact"]
