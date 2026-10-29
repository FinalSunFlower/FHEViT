from __future__ import annotations

import json

import pytest

from fpavlm.checkpoint import load_artifact_manifest, promote_artifact
from fpavlm.provenance import files_fingerprint, make_provenance, validate_provenance


def test_file_fingerprint_binds_path_and_content(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("same", encoding="utf-8")
    second.write_text("same", encoding="utf-8")
    both = files_fingerprint([first, second], root=tmp_path)
    first_only = files_fingerprint([first], root=tmp_path)
    assert both != first_only
    second.write_text("changed", encoding="utf-8")
    assert files_fingerprint([first, second], root=tmp_path) != both


def test_provenance_rejects_tampering():
    expected = make_provenance(purpose="unit-test", bindings={"artifact": "abc"})
    validate_provenance(expected, expected)
    tampered = dict(expected)
    tampered["bindings"] = {"artifact": "def"}
    with pytest.raises(ValueError, match="fingerprint"):
        validate_provenance(tampered, expected)


def test_promote_artifact_updates_manifest_atomically(tmp_path):
    manifest = tmp_path / "artifacts.manifest.json"
    artifact = tmp_path / "artifacts" / "model.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"checkpoint")
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "hash_algorithm": "sha256", "artifacts": {}}
        ),
        encoding="utf-8",
    )
    digest = promote_artifact(str(manifest), str(artifact))
    loaded = load_artifact_manifest(str(manifest))
    assert list(loaded.values()) == [digest]
