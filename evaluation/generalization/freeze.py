from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_freeze_manifest(
    repo_root: str | Path,
    *,
    baseline_sha: str,
) -> dict[str, object]:
    root = Path(repo_root)
    files = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted((root / "fabric_rlm").rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    return {
        "baseline_sha": baseline_sha,
        "scope": "fabric_rlm implementation and bundled skills",
        "files": files,
    }


def verify_freeze(
    repo_root: str | Path,
    manifest: Mapping[str, object],
) -> list[dict[str, str]]:
    root = Path(repo_root)
    expected = manifest.get("files")
    if not isinstance(expected, Mapping):
        raise ValueError("freeze manifest files must be an object")
    mismatches: list[dict[str, str]] = []
    for relative, digest in expected.items():
        path = root / str(relative)
        if not path.is_file():
            mismatches.append({"path": str(relative), "reason": "missing"})
        elif _sha256(path) != digest:
            mismatches.append({"path": str(relative), "reason": "changed"})
    current = {
        path.relative_to(root).as_posix()
        for path in (root / "fabric_rlm").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    for added in sorted(current - {str(path) for path in expected}):
        mismatches.append({"path": added, "reason": "added"})
    return mismatches


__all__ = ["create_freeze_manifest", "verify_freeze"]
