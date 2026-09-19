"""Offline checks: no Fabric authentication, semantic queries, or paid LM calls."""

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import fabric_rlm
import nbformat
import pytest


ROOT = Path(__file__).parents[1]
GENERATOR = ROOT / "examples/semantic_model/build_report_nb.py"
LEARN = ROOT / "examples/notebooks/rlm_learn_semantic_model_value.py"


@pytest.fixture
def report():
    spec = importlib.util.spec_from_file_location("semantic_report_example", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_builds_valid_clean_notebook(report, tmp_path, monkeypatch):
    # An import must not write the generated notebook into the shared worktree.
    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", Mock(side_effect=AssertionError("import wrote files")))
        spec = importlib.util.spec_from_file_location("import_only", GENERATOR)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))

    notebook = report.build_notebook()
    nbformat.validate(notebook)
    output = report.write_notebook(tmp_path / "report.Notebook")
    saved = nbformat.read(output / "notebook-content.ipynb", as_version=4)
    nbformat.validate(saved)
    assert [c.source for c in saved.cells] == [c.source for c in notebook.cells]
    assert json.loads((output / ".platform").read_text())["metadata"]["type"] == "Notebook"
    assert "dependencies" not in saved.metadata
    assert sum("parameters" in c.metadata.get("tags", []) for c in saved.cells) == 1
    for index, cell in enumerate(saved.cells):
        if cell.cell_type == "code":
            assert cell.outputs == []
            assert cell.execution_count is None
            if not cell.source.startswith("%pip"):
                compile(cell.source, f"report-cell-{index}", "exec")


def test_report_setup_and_task_use_fabric_lm(report, monkeypatch, capsys):
    lm = SimpleNamespace(model="azure/gpt-5.1", kwargs={"api_key": "test-secret"})
    factory = Mock(return_value=lm)
    semantic_model = Mock()
    task = Mock()
    monkeypatch.setitem(sys.modules, "fabric_rlm", SimpleNamespace(FabricLM=factory))
    namespace = {}
    exec(report.CELL_PARAM, namespace)
    with pytest.raises(ValueError, match="WORKSPACE_ID and MODEL_NAME"):
        exec(report.CELL_SETUP, namespace)
    factory.assert_not_called()

    namespace.update(WORKSPACE_ID="test-workspace", MODEL_NAME="test-model")
    exec(report.CELL_SETUP, namespace)
    factory.assert_called_once_with("gpt-5.1")
    assert namespace["LM"] is lm
    exec(report.CELL_TASK, namespace)
    namespace.update(
        RLM=SimpleNamespace(task=task), SemanticModel=semantic_model,
        OUT=report.OUT_XLSX, validate=Mock(), skills=["report_context"], loader=Mock(),
    )
    # Execute the real task construction, not the file deletion or live run cell.
    assignment = next(
        node for node in ast.parse(report.CELL_RUN).body
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "r" for target in node.targets
        )
    )
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "task", "exec"), namespace)
    assert task.call_args.kwargs["lm"] is lm
    assert task.call_args.kwargs["inputs"]["model"] is semantic_model.return_value
    semantic_model.assert_called_once_with("test-model", workspace="test-workspace")
    assert "test-secret" not in capsys.readouterr().out

    tree = ast.parse(report.CELL_TRUTH)
    queries = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("evaluate_dax", "list_columns")
    ]
    assert len(queries) == 2
    assert all(any(
        kw.arg == "workspace" and isinstance(kw.value, ast.Name)
        and kw.value.id == "WORKSPACE_ID" for kw in query.keywords
    ) for query in queries)


def test_learn_cells_compile_and_lm_settings_are_shared(capsys):
    source = LEARN.read_text(encoding="utf-8")
    cells = source.split("# CELL ********************")[1:]
    for index, cell in enumerate(cells):
        if not cell.lstrip().startswith("%pip"):
            compile(cell, f"learn-cell-{index}", "exec")
    tree = ast.parse("\n".join(line for line in source.splitlines() if not line.startswith("%")))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)]
    namespace = {"Path": Path}
    config_names = {
        "WORKSPACE_ID", "MODEL_ID", "MEASURE", "QUESTION", "LM_MODEL",
        "REASONING_EFFORT", "MAX_TURNS", "KNOWLEDGE_STORE",
    }
    config = [node for node in assignments if any(
        isinstance(target, ast.Name) and target.id in config_names for target in node.targets
    )]
    exec(compile(ast.Module(body=config, type_ignores=[]), "config", "exec"), namespace)
    assert namespace["WORKSPACE_ID"] == "<workspace-id>"
    assert namespace["MODEL_ID"] == "<semantic-model-id>"
    assert namespace["MEASURE"] in namespace["QUESTION"]

    factory = Mock(return_value=SimpleNamespace(model="azure/gpt-5.1", kwargs={}))
    namespace["FabricLM"] = factory
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "FabricLM"]
    assert len(calls) == 2
    for call in calls:
        eval(compile(ast.Expression(call), "lm-setup", "eval"), namespace)
    assert all(call.args == ("gpt-5.1",) and call.kwargs == {
        "reasoning_effort": "medium", "cache": False,
    } for call in factory.call_args_list)
    assert capsys.readouterr().out == ""


def test_examples_have_release_pins_and_no_embedded_resources(report):
    sources = ["\n".join(c.source for c in report.build_notebook().cells),
               LEARN.read_text(encoding="utf-8")]
    for source in sources:
        assert re.findall(r"fabric-rlm(?:\[[^\]]+\])?==([\d.]+)", source) == [fabric_rlm.__version__]
        assert not re.search(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", source, re.I)
        assert "orkey.txt" not in source
        assert "openrouter" not in source.lower()
        assert "api_key" not in source
