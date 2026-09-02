from __future__ import annotations

from pathlib import Path


def test_deprecated_notebook_utility_namespace_is_absent() -> None:
    repository = Path(__file__).resolve().parents[1]
    deprecated = "ms" + "sparkutils"
    roots = (
        repository / "fabric_rlm",
        repository / "examples",
        repository / "tests",
        repository / ".github",
    )
    candidates = [repository / "README.md", repository / "SECURITY.md"]
    for root in roots:
        candidates.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.casefold()
            in {".py", ".md", ".ipynb", ".json", ".toml", ".yaml", ".yml"}
            and "__pycache__" not in path.parts
        )

    matches = [
        str(path.relative_to(repository))
        for path in candidates
        if deprecated in path.read_text(encoding="utf-8").casefold()
    ]

    assert matches == []
