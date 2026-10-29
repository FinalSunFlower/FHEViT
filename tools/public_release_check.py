"""Check that a source-only release does not contain local research assets."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ONLY = {
    ".venv", "build", "dist", "fpavlm.egg-info", "models", "data", "artifacts",
    "private-client", "experiments/results", "docs/handoff/work",
    "experiments/.cache", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".pyre",
    ".pytype", ".tox", ".nox", ".trae/cache", ".trae/tmp", ".ipynb_checkpoints",
    ".benchmarks", "cache", "logs", "runs", "tensorboard", "outputs", "results_tmp",
    "docs/handoff/DATA_REGISTRY.md", "docs/handoff/CONTAMINATION_REPORT.md",
    "docs/handoff/experiment_ledger.json", "docs/handoff/PROGRESS.md",
    "docs/handoff/FINAL_REPORT.md",
    "figures/FIGURE1_PROMPT.md", "figures/figure1_system_architecture.png",
}
PUBLIC_INDEX_FILES = {
    "data/README.md", "data/README.zh-CN.md", "models/README.md",
    "artifacts/README.md", "experiments/results/README.md",
}
INDEXED_LOCAL_DIRS = {"data", "models", "artifacts", "experiments/results"}
TEXT_SUFFIXES = {".c", ".cfg", ".h", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yml", ".yaml"}
PATTERNS = {
    "absolute_windows_path": re.compile(r"\b[A-Za-z]:\\(?:Users|Documents|desktop|home)\\", re.I),
    "absolute_posix_path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "access_token": re.compile(r"\b(?:hf|sk)-[A-Za-z0-9_-]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def local_only(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    if relative in PUBLIC_INDEX_FILES:
        return False
    return any(relative == item or relative.startswith(item + "/") for item in LOCAL_ONLY)


def main() -> int:
    findings = []
    checked = 0
    for base, directories, filenames in os.walk(ROOT):
        base_path = Path(base)
        directories[:] = [name for name in directories if name not in {".git", "__pycache__"}
                          and (not local_only(base_path / name)
                               or (base_path / name).relative_to(ROOT).as_posix() in INDEXED_LOCAL_DIRS)]
        for filename in filenames:
            path = base_path / filename
            if local_only(path) or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            checked += 1
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                findings.append((path.relative_to(ROOT).as_posix(), "non_utf8_text"))
                continue
            for name, pattern in PATTERNS.items():
                if pattern.search(text):
                    findings.append((path.relative_to(ROOT).as_posix(), name))
    git_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if git_root.returncode == 0:
        repository = Path(git_root.stdout.strip()).resolve()
        # A cloned project can be nested in a larger Git worktree. Use the
        # actual repository root reported by Git, but avoid treating a nested
        # temporary test repository as the parent workspace.
        if not (repository / ".git").exists() and (ROOT / ".git").exists():
            repository = ROOT.resolve()
        try:
            prefix = ROOT.resolve().relative_to(repository).as_posix()
        except ValueError:
            prefix = ""
        if prefix == ".":
            prefix = ""
        tracked = subprocess.run(
            ["git", "ls-files", "--full-name", "-z"], cwd=repository,
            capture_output=True, check=False,
        )
        if tracked.returncode == 0:
            for entry in tracked.stdout.decode("utf-8", errors="replace").split("\0"):
                if not entry:
                    continue
                if prefix:
                    if not entry.startswith(prefix + "/"):
                        continue
                    relative = entry[len(prefix) + 1:]
                else:
                    relative = entry
                path = ROOT / Path(relative)
                if local_only(path):
                    findings.append((relative, "local_only_file_in_git_index"))
    if findings:
        for path, reason in findings:
            print(f"BLOCKED {reason}: {path}")
        return 1
    print(f"public source check passed ({checked} text files; local asset trees excluded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
