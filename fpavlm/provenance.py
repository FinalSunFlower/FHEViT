"""Deterministic provenance records for research caches and learned artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Union

PathLike = Union[str, Path]
PROVENANCE_SCHEMA_VERSION = 1


def file_sha256(path: PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_fingerprint(paths: Iterable[PathLike], *, root: PathLike) -> str:
    """Bind ordered relative paths and their contents into one SHA-256 digest."""
    root_path = Path(root).resolve()
    digest = hashlib.sha256()
    resolved = sorted((Path(path).resolve() for path in paths), key=lambda p: p.as_posix())
    if not resolved:
        raise ValueError("cannot fingerprint an empty file collection")
    for path in resolved:
        try:
            relative = path.relative_to(root_path).as_posix()
        except ValueError as exc:
            raise ValueError(f"provenance file lies outside root: {path}") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def source_tree_fingerprint(root: PathLike) -> str:
    root_path = Path(root).resolve()
    paths = list((root_path / "fpavlm").rglob("*.py"))
    paths.extend((root_path / "experiments").glob("*.py"))
    return files_fingerprint(paths, root=root_path)


def directory_fingerprint(root: PathLike) -> str:
    root_path = Path(root).resolve()
    paths = [
        path for path in root_path.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    ]
    return files_fingerprint(paths, root=root_path)


def python_source_fingerprint(root: PathLike) -> str:
    root_path = Path(root).resolve()
    paths = [
        path for path in root_path.rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    if not paths:
        raise ValueError(f"no Python sources found under {root_path}")
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        relative = path.relative_to(root_path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_provenance(*, purpose: str, bindings: Mapping[str, Any]) -> dict[str, Any]:
    if not purpose.strip():
        raise ValueError("provenance purpose must be non-empty")
    record = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "purpose": purpose,
        "bindings": dict(bindings),
    }
    record["fingerprint"] = canonical_fingerprint(record)
    return record


def validate_provenance(record: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("artifact provenance is missing")
    if record.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("unsupported provenance schema")
    without_fingerprint = {key: value for key, value in record.items() if key != "fingerprint"}
    if record.get("fingerprint") != canonical_fingerprint(without_fingerprint):
        raise ValueError("provenance fingerprint is invalid")
    if dict(record) != dict(expected):
        raise ValueError("artifact provenance does not match the current experiment inputs")


__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "canonical_fingerprint",
    "directory_fingerprint",
    "file_sha256",
    "files_fingerprint",
    "make_provenance",
    "python_source_fingerprint",
    "source_tree_fingerprint",
    "validate_provenance",
]
