from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_file_destination_readme_example_is_self_contained() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", readme, re.DOTALL)
    example = next(
        (block for block in blocks if "with FileDestination(" in block),
        None,
    )

    assert example is not None, "No FileDestination code block found in README"
    assert "LakehouseSource" in example
    assert "lakehouse = LakehouseSource(" in example
    compile(example, "<README FileDestination example>", "exec")


def test_project_uses_modern_spdx_license_metadata() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires = ["setuptools>=77", "wheel"]' in pyproject
    assert re.search(r'^license = "MIT"$', pyproject, re.MULTILINE)
    assert "License :: OSI Approved :: MIT License" not in pyproject
