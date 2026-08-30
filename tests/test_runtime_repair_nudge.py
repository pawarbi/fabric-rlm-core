"""Repair-turn diversity nudge (reviewer item: "Nudge diversity on repair turns").

Pins the three modes of ``FABRIC_RLM_REPAIR_NUDGE`` that gate the diversity line
appended to repair feedback:

* ``off``        -> line never appears (pre-0.2.x baseline).
* ``static``     -> line appears on every repair turn (reviewer's version).
* ``escalating`` -> line is suppressed on the first failure of a given repair key
                    and appears from the second failure onward (repeat-aware;
                    the default mode).

The three repair messages (output-validation, skill verifier, output validator)
all route through ``RLM._repair_nudge_suffix``; these tests exercise the shared
gate directly plus one formatter end-to-end.
"""

from __future__ import annotations

import pytest

from fabric_rlm import RLM
from fabric_rlm.interpreter import ExecResult
from fabric_rlm.runtime import _REPAIR_DIVERSITY_LINE, OutputValidationResult


def _make_rlm() -> RLM:
    def _lm(*, messages):  # never called in these tests
        raise AssertionError("LM should not be called")

    return RLM.from_task(
        task="dummy",
        inputs={},
        outputs=["answer"],
        lm=_lm,
        max_turns=3,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FABRIC_RLM_REPAIR_NUDGE", raising=False)
    yield


def test_off_never_appends(monkeypatch):
    monkeypatch.setenv("FABRIC_RLM_REPAIR_NUDGE", "off")
    rlm = _make_rlm()
    assert rlm._repair_nudge_suffix("k") == ""
    assert rlm._repair_nudge_suffix("k") == ""


def test_static_always_appends(monkeypatch):
    monkeypatch.setenv("FABRIC_RLM_REPAIR_NUDGE", "static")
    rlm = _make_rlm()
    first = rlm._repair_nudge_suffix("k")
    second = rlm._repair_nudge_suffix("k")
    assert _REPAIR_DIVERSITY_LINE in first
    assert _REPAIR_DIVERSITY_LINE in second


def test_escalating_suppresses_first_appends_second(monkeypatch):
    monkeypatch.setenv("FABRIC_RLM_REPAIR_NUDGE", "escalating")
    rlm = _make_rlm()
    assert rlm._repair_nudge_suffix("k") == ""
    assert _REPAIR_DIVERSITY_LINE in rlm._repair_nudge_suffix("k")
    assert _REPAIR_DIVERSITY_LINE in rlm._repair_nudge_suffix("k")


def test_escalating_keys_tracked_independently(monkeypatch):
    monkeypatch.setenv("FABRIC_RLM_REPAIR_NUDGE", "escalating")
    rlm = _make_rlm()
    assert rlm._repair_nudge_suffix("a") == ""
    assert rlm._repair_nudge_suffix("b") == ""  # different key -> still first failure
    assert _REPAIR_DIVERSITY_LINE in rlm._repair_nudge_suffix("a")


def test_unset_default_is_escalating(monkeypatch):
    monkeypatch.delenv("FABRIC_RLM_REPAIR_NUDGE", raising=False)
    rlm = _make_rlm()
    assert rlm._repair_nudge_suffix("k") == ""  # first failure: plain
    assert _REPAIR_DIVERSITY_LINE in rlm._repair_nudge_suffix("k")  # 2nd+: nudge


def test_unknown_mode_defaults_escalating(monkeypatch):
    monkeypatch.setenv("FABRIC_RLM_REPAIR_NUDGE", "bogus")
    rlm = _make_rlm()
    assert rlm._repair_nudge_suffix("k") == ""
    assert _REPAIR_DIVERSITY_LINE in rlm._repair_nudge_suffix("k")


def test_validation_feedback_respects_mode(monkeypatch):
    rlm = _make_rlm()
    result = ExecResult(
        ok=True, submitted=True, stdout="", stderr="",
        state={}, error=None, submit_payload={"answer": ""},
    )
    validation = OutputValidationResult(errors=("answer is blank",))

    monkeypatch.setenv("FABRIC_RLM_REPAIR_NUDGE", "off")
    assert _REPAIR_DIVERSITY_LINE not in rlm._format_validation_feedback(result, 1, validation)

    monkeypatch.setenv("FABRIC_RLM_REPAIR_NUDGE", "static")
    rlm._repair_counts = {}
    assert _REPAIR_DIVERSITY_LINE in rlm._format_validation_feedback(result, 1, validation)

    monkeypatch.setenv("FABRIC_RLM_REPAIR_NUDGE", "escalating")
    rlm._repair_counts = {}
    assert _REPAIR_DIVERSITY_LINE not in rlm._format_validation_feedback(result, 1, validation)
    assert _REPAIR_DIVERSITY_LINE in rlm._format_validation_feedback(result, 2, validation)


def test_name_error_for_json_null_explains_python_literals():
    rlm = _make_rlm()
    result = ExecResult(
        ok=False,
        submitted=False,
        stdout="",
        stderr="",
        state={"analysis": []},
        error="NameError: name 'null' is not defined",
        submit_payload=None,
    )

    feedback = rlm._format_feedback(result, 3)

    assert "use `None`, `True`, and `False`" in feedback
    assert "not JSON `null`, `true`, or `false`" in feedback


def test_unrelated_name_error_omits_json_literal_hint():
    rlm = _make_rlm()
    result = ExecResult(
        ok=False,
        submitted=False,
        stdout="",
        stderr="",
        state={},
        error="NameError: name 'revenue' is not defined",
        submit_payload=None,
    )

    assert "not JSON `null`, `true`, or `false`" not in rlm._format_feedback(result, 3)
