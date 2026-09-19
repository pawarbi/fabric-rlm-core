"""Offline notebook checks, not validation of Fabric auth or live model quality.

Execute the shipped cells with only the Lakehouse root redirected to tmp_path
and FabricLM replaced by scripted responses. RLM and its workers remain real.
"""

import ast
import csv
import io
import json
from pathlib import Path
import re
import socket
from types import SimpleNamespace

import fabric_rlm
import pytest

from test_api_arguments_engines import _ScriptedLM


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "notebooks"
LAKEHOUSE_ROOT = "/lakehouse/default/Files/fabric_rlm_examples"
NAMES = ("rlm_api_tour.ipynb", "rlm_spark_log_root_cause.ipynb")


def _cells(name, kind="code"):
    notebook = json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == kind]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")

    def no_connect(*args, **kwargs):
        raise AssertionError("Notebook tests must not make network calls")

    monkeypatch.setattr(socket.socket, "connect", no_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", no_connect)


@pytest.mark.parametrize("name", NAMES)
def test_starter_setup_and_python_cells(name):
    notebook = json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = _cells(name)
    assert code[0].startswith("%pip install")
    assert f"fabric-rlm=={fabric_rlm.__version__}" in code[0]
    assert "--no-deps" not in code[0]
    prose = "\n".join(_cells(name, "markdown"))
    assert "restart the Python session" in prose
    assert "default Lakehouse attached" in prose
    assert "Offline tests" in prose
    assert LAKEHOUSE_ROOT in "\n".join(code)
    for source in code[1:]:
        tree = ast.parse(source)
        assert not any(isinstance(node, ast.Try) for node in ast.walk(tree))
        assert "Path.cwd()" not in source
        assert ".exists()" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def _run_cells(name, tmp_path, monkeypatch, codes):
    lm = _ScriptedLM(codes)
    monkeypatch.setattr(fabric_rlm, "FabricLM", lambda *args, **kwargs: lm)
    namespace = {}
    for index, source in enumerate(_cells(name)[1:], start=1):
        # Replace only the prescribed root; all fixture and task code is shipped code.
        source = source.replace(LAKEHOUSE_ROOT, tmp_path.as_posix())
        exec(compile(source, f"{name}:cell {index}", "exec"), namespace)
    assert not lm.codes, "Some notebook recipes did not run"
    return namespace


def test_api_tour_runs_with_real_worker_and_typed_repair(tmp_path, monkeypatch):
    prose = "\n".join(_cells(NAMES[0], "markdown"))
    typed_submit = re.search(r"```python\n(SUBMIT\(.*?)\n```", prose, re.DOTALL).group(1)
    ns = _run_cells(NAMES[0], tmp_path, monkeypatch, [
        "SUBMIT(total=sum(numbers))",
        'SUBMIT(largest=str(max(sales)), average=str(sum(sales) / len(sales)))',
        typed_submit,
        "import csv, io\n"
        "rows = list(csv.DictReader(io.StringIO(sales_file.read_text())))\n"
        "totals = {}\n"
        "for row in rows:\n"
        "    totals[row['region']] = totals.get(row['region'], 0.0) + float(row['revenue'])\n"
        "SUBMIT(top_region=max(totals, key=totals.get), total_revenue=float(sum(totals.values())))",
        "import csv, io\n"
        "SUBMIT(row_count=len(list(csv.DictReader(io.StringIO(sales_file.read_text())))))",
    ])
    assert ns["result"].outputs == {"largest": 240, "average": 125.0}
    assert type(ns["result"].outputs["largest"]) is int
    assert type(ns["result"].outputs["average"]) is float
    assert len(ns["result"].turns) == 2  # Numeric strings were rejected before repair.
    assert ns["file_result"].outputs == {"top_region": "North", "total_revenue": 1175.0}
    assert type(ns["file_result"].outputs["total_revenue"]) is float
    assert ns["secure_result"].outputs == {"row_count": 4}
    rows = list(csv.DictReader(io.StringIO(ns["csv_file"].read_text())))
    assert len(rows) == 4
    assert sum(int(row["revenue"]) for row in rows) == 1175
    assert sum(int(row["revenue"]) for row in rows if row["region"] == "North") == 500
    assert max(rows, key=lambda row: int(row["revenue"]))["region"] == "South"
    assert ns["skill_loader"].load("regional_sales_rules") == ns["custom_skill"]
    assert "data_exploration" in ns["skill_loader"].list_skills()
    fixture = next(source for source in _cells(NAMES[0]) if "source_dir.mkdir" in source)
    before = ns["csv_file"].read_text()
    exec(fixture.replace(LAKEHOUSE_ROOT, tmp_path.as_posix()), ns)
    assert ns["csv_file"].read_text() == before


def test_spark_log_runs_and_checks_grounded_evidence(tmp_path, monkeypatch, capsys):
    ns = _run_cells(NAMES[1], tmp_path, monkeypatch, [
        "import re\n"
        "lines = log_file.read_text().splitlines()\n"
        "job = next(line for line in lines if re.search(r'Job \\d+ failed:', line))\n"
        "stage = next(line for line in lines if 'aborting job' in line)\n"
        "executor = next(line for line in lines if 'Lost executor' in line)\n"
        "SUBMIT(\n"
        "    failed_job_id=int(re.search(r'Job (\\d+) failed', job).group(1)),\n"
        "    failed_stage_id=int(re.search(r'stage (\\d+)\\.', stage).group(1)),\n"
        "    failing_executor=int(re.search(r'Lost executor (\\d+)', executor).group(1)),\n"
        "    root_cause=executor.split(': ', 2)[2],\n"
        "    evidence=[job, stage, executor],\n"
        ")",
    ])
    payload = ns["result"].outputs
    assert payload["failed_job_id"] == 17
    assert payload["failed_stage_id"] == 42
    assert payload["failing_executor"] == 7
    assert payload["root_cause"] == "java.lang.OutOfMemoryError: Java heap space"
    assert all(type(payload[key]) is int for key in (
        "failed_job_id", "failed_stage_id", "failing_executor",
    ))
    assert "Root cause:" in capsys.readouterr().out
    fixture = next(source for source in _cells(NAMES[1]) if "source_dir.mkdir" in source)
    before = ns["log_path"].read_text(encoding="utf-8")
    assert "Job 16 finished" in before
    assert before.count("WARN TaskSetManager: Lost task") == 4
    exec(fixture.replace(LAKEHOUSE_ROOT, tmp_path.as_posix()), ns)
    assert ns["log_path"].read_text(encoding="utf-8") == before
    check = _cells(NAMES[1])[-1]
    for incorrect in (
        {**payload, "failed_job_id": 16},
        {**payload, "evidence": ["invented evidence"]},
        {**payload, "evidence": [payload["evidence"][0]]},
    ):
        ns["result"] = SimpleNamespace(submitted=True, outputs=incorrect)
        with pytest.raises(AssertionError):
            exec(check, ns)
