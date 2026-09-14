"""The sweep, the brief, the reports and the KPIs stand on ``fabric_rlm.source_model`` alone.

The Data Agent review is experimental. Nothing on the supported path imports it,
and the old path ``fabric_rlm.data_agent_review`` still resolves every name.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import fabric_rlm

PACKAGE = Path(fabric_rlm.__file__).resolve().parent
FAMILY = ["source_model", "sweep", "brief", "reports", "kpis", "sweep_dashboard", "series"]
REVIEW_MODULES = ("data_agent_review", "experimental", "experimental.data_agent_review")


def test_the_sweep_family_imports_with_the_review_blocked():
    code = "\n".join(
        [
            "import sys",
            "sys.modules['fabric_rlm.experimental.data_agent_review'] = None",
            "sys.modules['fabric_rlm.data_agent_review'] = None",
            *[f"import fabric_rlm.{name}" for name in FAMILY],
            "print('ok')",
        ]
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(PACKAGE.parent), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


@pytest.mark.parametrize("name", FAMILY)
def test_the_sweep_family_never_imports_the_review(name):
    tree = ast.parse((PACKAGE / f"{name}.py").read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in REVIEW_MODULES or module.startswith("fabric_rlm.experimental") or module.endswith("data_agent_review"):
                offenders.append(f"line {node.lineno}: from {'.' * node.level}{module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "data_agent_review" in alias.name or "experimental" in alias.name:
                    offenders.append(f"line {node.lineno}: import {alias.name}")
    assert not offenders, offenders


def test_the_old_path_resolves_every_name_to_its_new_home():
    import fabric_rlm.data_agent_review as old
    from fabric_rlm import source_model
    from fabric_rlm.experimental import data_agent_review as review

    assert old.review_agent is review.review_agent
    assert old.SdkAgentReader is review.SdkAgentReader
    assert old.schema_from_tables is source_model.schema_from_tables
    assert old.LakehouseExecutor is source_model.LakehouseExecutor
    assert old._measure_columns is source_model._measure_columns
    assert list(old.__all__) == list(review.__all__)
    assert all(hasattr(review, name) for name in review.__all__)
    assert {"review_agent", "_measure_columns", "schema_from_tables"} <= set(dir(old))
    with pytest.raises(AttributeError):
        old.no_such_name  # noqa: B018


def test_the_source_model_does_not_reach_back_into_the_review():
    tree = ast.parse((PACKAGE / "source_model.py").read_text(encoding="utf-8"))
    modules = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    modules |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not {m for m in modules if "fabric_rlm" in m or m.startswith(".")}, modules
