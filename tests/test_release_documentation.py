from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
FALLBACK_BRANCH_INSTALL = (
    '"git+https://github.com/pawarbi/fabric-rlm-core.git@'
    'feature/knowledge-opportunistic-fallback"'
)


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


def test_project_uses_fabric_compatible_license_metadata() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires = ["setuptools>=77", "wheel"]' in pyproject
    assert re.search(r'^license = \{ text = "MIT" \}$', pyproject, re.MULTILINE)
    assert "License :: OSI Approved :: MIT License" not in pyproject


def test_analytics_extra_includes_semantic_model_dataframe_dependency() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    analytics = pyproject.split("analytics = [", 1)[1].split("]", 1)[0]

    assert '"pandas>=2.0"' in analytics


def test_runtime_dependencies_avoid_anyio_sentinel_import() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"anyio>=4.5,<4.15"' in pyproject
    assert '"typing_extensions>=4.15"' in pyproject
    assert '"numpy>=1.26,<2; python_version < \'3.13\'"' in pyproject


def test_fabric_setup_docs_use_the_dependency_resolving_fallback_install() -> None:
    for relative_path in (
        "README.md",
        "QUICKSTART.md",
        "docs/fabric-runtime-deps.md",
    ):
        document = (ROOT / relative_path).read_text(encoding="utf-8")

        assert FALLBACK_BRANCH_INSTALL in document
