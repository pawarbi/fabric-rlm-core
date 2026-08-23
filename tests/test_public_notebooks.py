from __future__ import annotations

import json
import re
from pathlib import Path

import fabric_rlm
import pytest


NOTEBOOK_DIR = Path(__file__).parents[1] / "examples" / "notebooks"


def _notebooks():
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        yield path, json.loads(path.read_text(encoding="utf-8"))


def _source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )


def test_public_notebooks_do_not_install_from_git_branches():
    offenders = [
        path.name
        for path, notebook in _notebooks()
        if "git+https://github.com/" in _source(notebook)
    ]

    assert offenders == []


def test_public_notebook_installs_pin_the_package_version():
    invalid_pins = []
    pattern = re.compile(r"fabric-rlm(?:\[[^\]]+\])?==([^\s\"']+)")

    for path, notebook in _notebooks():
        install_lines = [
            line
            for line in _source(notebook).splitlines()
            if line.lstrip().startswith("%pip install")
            and "fabric-rlm" in line
        ]
        for line in install_lines:
            match = pattern.search(line)
            if match is None:
                invalid_pins.append(f"{path.name}: unpinned")
            elif match.group(1) != fabric_rlm.__version__:
                invalid_pins.append(f"{path.name}: {match.group(1)}")

    assert invalid_pins == []


def test_public_notebook_release_metadata_matches_the_package_version():
    stale_metadata = []
    pattern = re.compile(r'GIT_COMMIT\s*=\s*"v([^"]+)"')

    for path, notebook in _notebooks():
        for version in pattern.findall(_source(notebook)):
            if version != fabric_rlm.__version__:
                stale_metadata.append(f"{path.name}: {version}")

    assert stale_metadata == []


@pytest.mark.parametrize(
    "notebook_name",
    [
        "spreadsheetbench_400_openrouter_minimax_mlflow.ipynb",
        "ssb400_minimax_m3_fabric_repro.ipynb",
    ],
)
def test_benchmark_notebooks_expose_a_limit_for_smoke_runs(notebook_name):
    notebook = json.loads(
        (NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8")
    )
    source = _source(notebook)

    assert "LIMIT = 400" in source
    assert "[:LIMIT]" in source


def test_ssb_repro_reads_dataset_json_as_utf8():
    notebook = json.loads(
        (
            NOTEBOOK_DIR / "ssb400_minimax_m3_fabric_repro.ipynb"
        ).read_text(encoding="utf-8")
    )

    assert (
        '(ds / "dataset.json").read_text(encoding="utf-8")'
        in _source(notebook)
    )


@pytest.mark.parametrize(
    "notebook_name",
    [
        "verify_block_network_fabric.ipynb",
        "verify_timeout_recovery_fabric.ipynb",
    ],
)
def test_release_verification_notebooks_install_the_release(notebook_name):
    notebook = json.loads(
        (NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert f"fabric-rlm=={fabric_rlm.__version__}" in code


@pytest.mark.parametrize(
    ("notebook_name", "failure_assertion"),
    [
        ("verify_block_network_fabric.ipynb", "assert passed == total"),
        ("verify_timeout_recovery_fabric.ipynb", "assert not failed"),
    ],
)
def test_release_verification_notebooks_fail_the_run_on_failed_checks(
    notebook_name, failure_assertion
):
    notebook = json.loads(
        (NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8")
    )
    final_code_cell = next(
        cell
        for cell in reversed(notebook["cells"])
        if cell.get("cell_type") == "code"
    )

    assert failure_assertion in "".join(final_code_cell["source"])


def test_public_notebooks_do_not_invite_keys_in_source():
    offenders = []
    literal_assignment = re.compile(
        r"(?mi)^\s*[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET)"
        r"\s*=\s*[\"'][^\"']*[\"']"
    )
    paste_instruction = re.compile(r"(?i)(paste.{0,20}key|key.{0,20}paste)")

    for path, notebook in _notebooks():
        source = _source(notebook)
        if literal_assignment.search(source) or paste_instruction.search(source):
            offenders.append(path.name)

    assert offenders == []


def test_api_tour_covers_the_practical_public_surface():
    notebook = json.loads(
        (NOTEBOOK_DIR / "rlm_api_tour.ipynb").read_text(encoding="utf-8")
    )
    source = _source(notebook)
    expected_examples = [
        f"fabric-rlm=={fabric_rlm.__version__}",
        'RLM("numbers -> total",',
        "RLM.task(",
        'outputs={"largest": int, "average": float}',
        "LocalArtifactStore(",
        "File(",
        'skills=["data_exploration"]',
        "output_validator=",
        "result.report(as_dict=True)",
        "result.turns",
        "from fabric_rlm.security import SecurityPolicy",
        "SecurityPolicy.default()",
        "block_network=True",
        "recover_worker_timeouts=1",
    ]

    missing = [example for example in expected_examples if example not in source]

    assert missing == []


def test_flagship_notebook_fails_on_failed_rlm_grades():
    notebook = json.loads(
        (NOTEBOOK_DIR / "rlm_vs_plain_llm_imf_cpi.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = _source(notebook)

    assert "assert all(checks.values())" in source
    assert "assert all(two_source_checks.values())" in source


def test_flagship_notebook_replaces_an_invalid_cached_pdf():
    notebook = json.loads(
        (NOTEBOOK_DIR / "rlm_vs_plain_llm_imf_cpi.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = _source(notebook)

    assert (
        "92eaa546c733eb98e63653d33be133dcf9c0a2b6"
        in source
    )
    assert (
        "cb0c2459c63b5d7d3d310fb6f0d6c32"
        "ee6624b79050dcf644e55a109a07bf360"
        in source.lower()
    )
    assert "WEO_MAX_BYTES" in source
    assert 'WEO_PATH + ".download"' in source
    assert "hashlib.sha256" in source
    assert "fitz.open" in source
    assert "os.replace" in source
    assert 'raise ValueError(f"Downloaded file is not a PDF: {WEO_PATH}")' in source


def test_network_verification_uses_the_live_lm_in_check_11():
    notebook = json.loads(
        (
            NOTEBOOK_DIR / "verify_block_network_fabric.ipynb"
        ).read_text(encoding="utf-8")
    )
    source = _source(notebook)

    assert "class LiveCheckedScriptedLM" in source
    assert "run_one(probe, lm=lm_for_probe" in source


def test_network_verification_reports_live_lm_errors_in_check_11():
    notebook = json.loads(
        (
            NOTEBOOK_DIR / "verify_block_network_fabric.ipynb"
        ).read_text(encoding="utf-8")
    )
    check_11 = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "run_one(probe, lm=lm_for_probe" in "".join(cell.get("source", []))
    )

    assert "    try:\n        r = run_one(probe, lm=lm_for_probe" in check_11
    assert "except Exception as e:" in check_11
    assert "credentials problem" in check_11
    assert (
        'check("11. worker still sealed in a run whose LM is reachable",'
        in check_11
    )


@pytest.mark.parametrize(
    "path,notebook",
    list(_notebooks()),
    ids=lambda value: value.name if isinstance(value, Path) else None,
)
def test_public_notebooks_use_a_supported_python_kernel(path, notebook):
    kernel_names = {
        notebook.get("metadata", {}).get("kernelspec", {}).get("name"),
        notebook.get("metadata", {}).get("kernel_info", {}).get("name"),
    } - {None}

    assert kernel_names, path.name
    assert kernel_names <= {"python3", "jupyter"}, (
        f"{path.name}: unsupported kernels {sorted(kernel_names)}"
    )
