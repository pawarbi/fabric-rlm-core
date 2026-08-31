from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "critic_evidence_closure.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "critic_evidence_closure", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_closure_audit_preserves_input_audit_when_skipped() -> None:
    closure = load_module()
    original_audit = {"checks": [{"path": "insights[0]", "actual": 1}]}

    resolved = closure.resolve_closure_audit(
        {"audit": None, "summary": {"skipped": True}},
        original_audit,
    )

    assert resolved is original_audit


def test_resolve_closure_audit_serializes_executed_audit() -> None:
    closure = load_module()
    report = object()
    serialized = {"checks": [{"path": "insights[0]", "actual": 2}]}
    closure._STAGED.audit_to_dict = lambda value: serialized if value is report else None

    resolved = closure.resolve_closure_audit(
        {"audit": report, "summary": {"skipped": False}},
        {"checks": []},
    )

    assert resolved == serialized
