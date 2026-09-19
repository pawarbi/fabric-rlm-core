"""Keep public examples focused and free of embedded credentials."""

import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted(
    path for path in (ROOT / "examples").rglob("*")
    if path.is_file() and path.suffix in {".py", ".ipynb", ".md"}
    and "__pycache__" not in path.parts
)


def test_deterministic_what_moved_notebook_is_not_a_public_rlm_example():
    for suffix in (".py", ".ipynb"):
        assert not (ROOT / "examples/notebooks" / ("rlm_what_moved" + suffix)).exists()
    for name in ("examples/README.md", "README.md", "docs/usage-guide.md"):
        assert "rlm_what_moved" not in (ROOT / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.relative_to(ROOT).as_posix())
def test_public_example_scope(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        # Decode JSON escapes too, including markdown split across source lines.
        text += "\n" + "\n".join(
            "".join(cell.get("source", "")) for cell in json.loads(text)["cells"]
        )
    text = re.sub(r"\n\s*#\s?", "\n", text)
    assert not re.search(r"data[_\s-]?agents?(?!bench)", path.name + "\n" + text, re.I), (
        "Public examples must not contain Data Agent recipes or references"
    )


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.relative_to(ROOT).as_posix())
def test_public_examples_do_not_embed_credentials(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        text += "\n" + "\n".join(
            "".join(cell.get("source", "")) for cell in json.loads(text)["cells"]
        )
    forbidden = (
        r"\bsk-(?:proj-|or-v1-|ant-)?[A-Za-z0-9_-]{16,}",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r'''\b(?:api[_-]?key|access[_-]?token|client[_-]?secret)["']?\s*[:=]\s*["'][^"'<>\s]{8,}["']''',
        r"(?:[/\\]|\b)(?:orkey|[\w-]*(?:api[_-]?key|secret|credential)[\w-]*)\.(?:txt|json|ya?ml)\b",
    )
    for pattern in forbidden:
        assert not re.search(pattern, text, re.I), (
            f"Embedded credential or plaintext secret-file path in {path.relative_to(ROOT)}"
        )


@pytest.mark.parametrize(
    "path",
    [path for path in EXAMPLES if path.suffix == ".ipynb"],
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_public_notebook_code_compiles_and_is_unexecuted(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        location = f"{path.relative_to(ROOT)}:cell{index}"
        code = "\n".join(
            line for line in "".join(cell["source"]).splitlines()
            if not line.lstrip().startswith("%")
        )
        compile(code, location, "exec")
        assert cell["outputs"] == [], location
        assert cell["execution_count"] is None, location
