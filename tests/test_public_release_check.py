from __future__ import annotations

import re
import subprocess

from tools import public_release_check


def test_public_data_indexes_are_scanned_but_payloads_are_excluded(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    data.mkdir()
    (data / "README.md").write_text("INDEX_CANARY", encoding="utf-8")
    (data / "README.zh-CN.md").write_text("INDEX_CANARY", encoding="utf-8")
    (data / "patient.csv").write_text("INDEX_CANARY", encoding="utf-8")
    monkeypatch.setattr(public_release_check, "ROOT", tmp_path)
    monkeypatch.setattr(public_release_check, "PATTERNS", {"test_marker": re.compile("INDEX_CANARY")})

    assert public_release_check.main() == 1

    output = capsys.readouterr().out
    assert "BLOCKED test_marker: data/README.md" in output
    assert "BLOCKED test_marker: data/README.zh-CN.md" in output
    assert "patient.csv" not in output


def test_forced_git_add_of_local_data_is_blocked(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    data.mkdir()
    (data / "patient.csv").write_text("local only", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "--force", "data/patient.csv"], cwd=tmp_path, check=True)
    monkeypatch.setattr(public_release_check, "ROOT", tmp_path)

    assert public_release_check.main() == 1

    assert "BLOCKED local_only_file_in_git_index: data/patient.csv" in capsys.readouterr().out
