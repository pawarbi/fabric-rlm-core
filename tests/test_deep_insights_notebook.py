"""Execute the shipped recipe offline; these tests do not audit a live model."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import re
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import fabric_rlm
import IPython.display
import pytest

from fabric_rlm.skill_loader import SkillLoader
from test_deep_insight_discovery_skill import (
    _derived_metric_spec,
    _diagnostic_assessment,
    _metric_component,
    _strong_candidates,
    _strong_insight,
    _strong_plan,
)


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "examples/notebooks/rlm_verified_deep_insights.ipynb"
)


@pytest.fixture
def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def no_connect(*args, **kwargs):
        raise AssertionError("Notebook tests must not make external calls")

    monkeypatch.setattr(socket.socket, "connect", no_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", no_connect)


@pytest.fixture
def valid_payload():
    # Reuse the skill's version-2 fixture, not a purported scrap-model answer.
    insight = _strong_insight()
    insight["diagnostic_measurability"] = "mixed"
    insight["diagnostic_assessment"] = _diagnostic_assessment(
        measurable=True, disposition="ruled_out", decision_readiness="act_ready"
    )
    insight["action"]["kind"] = "program"
    insight["metric_spec"] = _derived_metric_spec(
        "delta", 13,
        [
            _metric_component("current", "current", 78),
            _metric_component("comparison", "comparison", 65),
        ],
    )
    return {
        "contract_version": 2,
        "analysis_plan": _strong_plan(),
        "candidates": _strong_candidates([insight]),
        "insights": [insight],
        "report_markdown": "## Findings\n\n" + insight["statement"],
    }


def test_notebook_setup_and_metadata(notebook):
    assert (notebook["nbformat"], notebook["nbformat_minor"]) == (4, 5)
    cells = notebook["cells"]
    assert 6 <= len(cells) <= 7
    ids = [cell["id"] for cell in cells]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) for value in ids)
    metadata = notebook["metadata"]
    assert metadata["kernelspec"]["name"] == "python3"
    assert metadata["language_info"]["version"] == "3.12"
    assert metadata["kernel_info"] == {
        "name": "jupyter", "jupyter_kernel_name": "python3.12",
    }
    assert metadata["microsoft"]["language_group"] == "jupyter_python"
    assert "dependencies" not in metadata
    assert "".join(cells[1]["source"]) == (
        f'%pip install "fabric-rlm[analytics]=={fabric_rlm.__version__}"'
    )
    assert cells[2]["cell_type"] == "markdown"
    assert "restart the python session" in "".join(cells[2]["source"]).lower()
    prose = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
    ).lower()
    for requirement in (
        "python 3.12", "semantic-link-sempy", "read/query permissions",
        "no lakehouse is required", "structure/arithmetic checks only",
        "not an automatic independent host-side", "markdown prose is not verified",
        "verifier errors may degrade", "inspect the trace",
    ):
        assert requirement in prose
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        source = "".join(cell["source"])
        if source.startswith("%pip"):
            continue
        tree = ast.parse(source)
        assert not any(isinstance(node, ast.Try) for node in ast.walk(tree))
        assert "credential_provider" not in source
        assert "api_key" not in source.lower()
        assert "verbose" not in source


@pytest.mark.parametrize("submitted", [True, False])
def test_notebook_executes_task_and_displays_only_submitted_report(
    notebook, monkeypatch, valid_payload, submitted
):
    result = SimpleNamespace(
        submitted=submitted,
        outputs=valid_payload,
        report=Mock(return_value="No submission: discovery failed"),
        trajectory=SimpleNamespace(metadata={"verifier_execution": {"status": "passed"}}),
    )
    semantic_model = Mock(return_value=object())
    fabric_lm = Mock(return_value=object())
    rlm = Mock()
    rlm.task.return_value.run.return_value = result
    display = Mock()
    monkeypatch.setattr(fabric_rlm, "SemanticModel", semantic_model)
    monkeypatch.setattr(fabric_rlm, "FabricLM", fabric_lm)
    monkeypatch.setattr(fabric_rlm, "RLM", rlm)
    monkeypatch.setattr(IPython.display, "display", display)
    namespace = {}
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or source.startswith("%pip"):
            continue
        code = compile(source, f"{NOTEBOOK.name}:{cell['id']}", "exec")
        if cell["id"] == "report" and not submitted:
            with pytest.raises(AssertionError, match="No submission: discovery failed"):
                exec(code, namespace)
        else:
            exec(code, namespace)

    semantic_model.assert_called_once_with(
        "<semantic-model-id>", workspace="<workspace-id>"
    )
    fabric_lm.assert_called_once_with("gpt-5.1", reasoning_effort="high")
    rlm.task.assert_called_once()
    rlm.task.return_value.run.assert_called_once_with()
    kwargs = rlm.task.call_args.kwargs
    assert kwargs["inputs"] == {"source": semantic_model.return_value}
    assert kwargs["lm"] is fabric_lm.return_value
    assert kwargs["enable_verifier"] is True
    assert kwargs["max_turns"] == 25
    assert kwargs["outputs"] == {
        "contract_version": int, "analysis_plan": dict, "candidates": list,
        "insights": list, "report_markdown": str,
    }
    assert all(type(valid_payload[key]) is kind for key, kind in kwargs["outputs"].items())
    assert kwargs["skills"] == [
        "semantic_model", "analytical_integrity", "deep_insight_discovery",
    ]
    for name in kwargs["skills"]:
        assert SkillLoader().load(name).content
    task = " ".join(kwargs["task"].lower().split())
    for requirement in (
        "governed measures", "definitions, relationships, and available periods",
        "latest two complete periods", "source coverage",
        "if completeness cannot be confirmed, disclose",
        "plant and product category", "within-group rate changes", "mix changes",
        "execute independent source queries", "explicit residual",
        "execute the verification expressions", "contract version 2",
        "at most 3 actual candidates", "do not invent candidates",
        "do not invent scrap reasons, causal explanations, or materiality thresholds",
        "report_markdown", "faithful rendering", "tables", "limitations",
        "no extra numbers or claims", "no code fences",
    ):
        assert requirement in task
    assert namespace["result"] is result
    if submitted:
        result.report.assert_not_called()
        display.assert_called_once()
        rendered = display.call_args.args[0]
        assert isinstance(rendered, IPython.display.Markdown)
        assert rendered.data == valid_payload["report_markdown"]
    else:
        result.report.assert_called_once_with()
        display.assert_not_called()


def test_report_markdown_preserves_skill_verifier_checks(valid_payload):
    source = SkillLoader().load("deep_insight_discovery").verifier_source
    assert source is not None
    namespace = {}
    exec(compile(source, "deep_insight_discovery:verify", "exec"), namespace)
    verify = namespace["verify"]
    verify(valid_payload)
    without_report = deepcopy(valid_payload)
    del without_report["report_markdown"]
    verify(without_report)
    invalid = deepcopy(valid_payload)
    invalid["insights"][0]["metric_spec"]["expected_value"] += 1
    with pytest.raises(AssertionError, match="does not reconcile"):
        verify(invalid)
    invalid = deepcopy(valid_payload)
    del invalid["insights"][0]["diagnostic_assessment"]
    with pytest.raises(AssertionError, match="diagnostic"):
        verify(invalid)


def test_empty_insights_are_not_advertised_as_an_accepted_report(notebook, valid_payload):
    text = " ".join(" ".join("".join(c["source"]) for c in notebook["cells"]).split())
    assert "an empty findings list is rejected" in text
    assert "submit between one and three supported insights" in text
    assert "return fewer or no insights" not in text
    scope = {}
    exec(SkillLoader().load("deep_insight_discovery").verifier_source, scope)
    empty = deepcopy(valid_payload)
    empty["insights"] = []
    with pytest.raises(AssertionError):
        scope["verify"](empty)
