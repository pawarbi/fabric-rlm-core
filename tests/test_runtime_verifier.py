"""Tests for the runtime-enforced playbook verifier (item C)."""

from __future__ import annotations

from dataclasses import replace

import pytest

import fabric_rlm.runtime as runtime_mod
from fabric_rlm import RLM
from fabric_rlm.interpreter import ExecResult
from fabric_rlm.skill_loader import Skill, SkillLoader, extract_verifier_source


# --- Scripted-LM / FakeInterpreter scaffolding ----------------------------------


class ScriptedLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.messages: list[list[dict]] = []

    def __call__(self, *, messages):
        self.messages.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("No scripted LM responses left")
        return self.responses.pop(0)


class FakeInterpreter:
    """Stand-in for the real subprocess interpreter.

    Two-stage scripting: ``results`` is consumed for the LM-emitted code,
    ``verifier_results`` is consumed for verifier_code injected by the runtime
    (recognised via the ``_fabric_rlm_payload`` marker we bake into the
    verifier wrapper).
    """

    def __init__(self, results: list, verifier_results: list | None = None):
        self._results = list(results)
        self._verifier_results = list(verifier_results or [])
        self.executed: list[str] = []
        self.verifier_executed: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def configure_lm(self, spec):
        return {}

    def set_inputs(self, inputs):
        return None

    def execute(self, code: str):
        if "_fabric_rlm_payload" in code:
            self.verifier_executed.append(code)
            if not self._verifier_results:
                # Default: verifier passes silently.
                return ExecResult(ok=True, submitted=False, stdout="", stderr="", state={})
            item = self._verifier_results.pop(0)
            return item() if callable(item) else item
        self.executed.append(code)
        if not self._results:
            raise AssertionError("No scripted ExecResults left")
        item = self._results.pop(0)
        return item() if callable(item) else item


def _install_fake_interpreter(monkeypatch, results, verifier_results=None) -> FakeInterpreter:
    fake = FakeInterpreter(results, verifier_results)
    monkeypatch.setattr(runtime_mod, "Interpreter", lambda **kwargs: fake)
    return fake


def _submit(payload: dict, stdout: str = "") -> ExecResult:
    return ExecResult(
        ok=True,
        submitted=True,
        stdout=stdout,
        stderr="",
        state={},
        submit_payload=payload,
    )


def _ran(stdout: str = "") -> ExecResult:
    return ExecResult(ok=True, submitted=False, stdout=stdout, stderr="", state={})


def _verifier_pass() -> ExecResult:
    return ExecResult(ok=True, submitted=False, stdout="", stderr="", state={})


def _verifier_assert(message: str) -> ExecResult:
    return ExecResult(
        ok=False,
        submitted=False,
        stdout="",
        stderr="",
        state={},
        error=(
            "Traceback (most recent call last):\n"
            "  File \"<verifier>\", line 5, in <module>\n"
            "    verify(_fabric_rlm_payload)\n"
            "  File \"<verifier>\", line 3, in verify\n"
            f"AssertionError: {message}"
        ),
    )


def _verifier_name_error() -> ExecResult:
    return ExecResult(
        ok=False,
        submitted=False,
        stdout="",
        stderr="",
        state={},
        error=(
            "Traceback (most recent call last):\n"
            "  File \"<verifier>\", line 2, in verify\n"
            "NameError: name 'xyz' is not defined"
        ),
    )


# --- Stub skill plumbing (avoids depending on bundled markdown contents) ---------


def _stub_skill_loader(skills: dict[str, Skill]) -> SkillLoader:
    """Return a SkillLoader that yields the provided Skill objects by name."""

    loader = SkillLoader()

    def _fake_load(name: str) -> Skill:
        if name not in skills:
            raise FileNotFoundError(f"Unknown SKILL: {name}")
        return skills[name]

    loader.load = _fake_load  # type: ignore[method-assign]
    loader.format_index = lambda: ""  # type: ignore[method-assign]
    return loader


def _make_skill(name: str, verifier_source: str | None) -> Skill:
    return Skill(
        name=name,
        title=name,
        summary="",
        content=f"# {name}\nSummary: stub\n",
        dependencies=(),
        verifier_source=verifier_source,
    )


# --- Tests -----------------------------------------------------------------------


def test_skill_loader_extracts_verifier_source(tmp_path) -> None:
    """The loader pulls the first ``## Required verifier`` python block, or None."""

    md_with = (
        "# demo\nSummary: t\n\n"
        "## Required verifier\n\n"
        "```python\n"
        "def verify(payload):\n"
        "    assert payload['Q5'] == 0\n"
        "```\n\n"
        "## Common failure modes\n- foo\n"
    )
    md_without = "# demo\nSummary: t\n\n## Procedure\n1. Do the thing.\n"

    src = extract_verifier_source(md_with)
    assert src is not None
    assert src.startswith("def verify(payload):")
    assert "assert payload['Q5'] == 0" in src
    assert "```" not in src

    assert extract_verifier_source(md_without) is None
    assert extract_verifier_source("") is None

    # End-to-end via SkillLoader on a temp directory.
    (tmp_path / "demo_with.md").write_text(md_with, encoding="utf-8")
    (tmp_path / "demo_without.md").write_text(md_without, encoding="utf-8")
    loader = SkillLoader(skill_dir=tmp_path)
    with_verifier = loader.load("demo_with")
    without_verifier = loader.load("demo_without")
    assert with_verifier.verifier_source is not None
    assert "def verify(payload):" in with_verifier.verifier_source
    assert without_verifier.verifier_source is None


def test_verifier_passes_then_submit_returns(monkeypatch) -> None:
    """Valid SUBMIT -> verifier passes -> accept (no reflection turn)."""

    skill = _make_skill(
        "mcm_stub",
        "def verify(payload):\n    assert payload['Q5'] == (payload['Q4']-payload['Q3'])*payload['Q2']\n",
    )
    loader = _stub_skill_loader({"mcm_stub": skill})

    fake = _install_fake_interpreter(
        monkeypatch,
        results=[
            _submit({"Q1": "(M_1*M_2)", "Q2": 10, "Q3": 1, "Q4": 2, "Q5": 10}),
        ],
        verifier_results=[_verifier_pass()],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(Q1='(M_1*M_2)', Q2=10, Q3=1, Q4=2, Q5=10)\n```",
        ]
    )
    rlm = RLM.from_task(
        "Solve MCM.",
        outputs=["Q1", "Q2", "Q3", "Q4", "Q5"],
        lm=lm,
        max_turns=4,
        timeout=5,
        skills=["mcm_stub"],
        skill_loader=loader,
    )

    result = rlm.run()

    assert result.submitted is True
    assert result.payload == {"Q1": "(M_1*M_2)", "Q2": 10, "Q3": 1, "Q4": 2, "Q5": 10}
    turn_types = [t.turn_type for t in result.trajectory]
    assert turn_types == ["normal"]
    assert "verifier_repair" not in turn_types
    # The runtime ran exactly one verifier execution.
    assert len(fake.verifier_executed) == 1
    assert "verify(_fabric_rlm_payload)" in fake.verifier_executed[0]


def test_verifier_fails_triggers_repair_turn(monkeypatch) -> None:
    """Verifier raises AssertionError -> verifier_repair turn -> resubmit -> accept."""

    skill = _make_skill(
        "mcm_stub",
        "def verify(payload):\n    assert payload['Q5'] == 0, 'Q5 must equal (Q4-Q3)*Q2'\n",
    )
    loader = _stub_skill_loader({"mcm_stub": skill})

    fake = _install_fake_interpreter(
        monkeypatch,
        results=[
            _submit({"Q1": "(M_1*M_2)", "Q2": 10, "Q3": 1, "Q4": 2, "Q5": -99999}),
            _submit({"Q1": "(M_1*M_2)", "Q2": 10, "Q3": 1, "Q4": 2, "Q5": 10}),
        ],
        verifier_results=[
            _verifier_assert("Q5 must equal (Q4-Q3)*Q2 = 10, got -99999"),
            _verifier_pass(),
        ],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(Q1='(M_1*M_2)', Q2=10, Q3=1, Q4=2, Q5=-99999)\n```",
            "```python\nSUBMIT(Q1='(M_1*M_2)', Q2=10, Q3=1, Q4=2, Q5=10)\n```",
        ]
    )
    rlm = RLM.from_task(
        "Solve MCM.",
        outputs=["Q1", "Q2", "Q3", "Q4", "Q5"],
        lm=lm,
        max_turns=6,
        timeout=5,
        skills=["mcm_stub"],
        skill_loader=loader,
    )

    result = rlm.run()

    assert result.submitted is True
    assert result.payload["Q5"] == 10
    turn_types = [t.turn_type for t in result.trajectory]
    assert turn_types == ["normal", "verifier_repair"]
    # Repair feedback should reach the LM with the assertion message.
    second_call_messages = lm.messages[1]
    feedback = second_call_messages[-1]["content"]
    assert "rejected by the `mcm_stub` skill verifier" in feedback
    assert "Q5 must equal" in feedback
    assert "AssertionError" in feedback
    # Two verifier runs happened: failing then passing.
    assert len(fake.verifier_executed) == 2


def test_skill_without_verifier_no_change(monkeypatch) -> None:
    """Loading a skill without a verifier section behaves identically to pre-C."""

    skill = _make_skill("plain", verifier_source=None)
    loader = _stub_skill_loader({"plain": skill})

    fake = _install_fake_interpreter(
        monkeypatch,
        results=[
            _submit({"answer": 7}),
        ],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(answer=7)\n```",
        ]
    )
    rlm = RLM.from_task(
        "Return 7.",
        outputs=["answer"],
        lm=lm,
        max_turns=4,
        timeout=5,
        skills=["plain"],
        skill_loader=loader,
    )

    result = rlm.run()

    assert result.submitted is True
    assert result.payload == {"answer": 7}
    turn_types = [t.turn_type for t in result.trajectory]
    assert turn_types == ["normal"]
    # No verifier code was ever executed.
    assert fake.verifier_executed == []


def test_buggy_verifier_logs_and_accepts(monkeypatch, caplog) -> None:
    """A verifier raising NameError is treated as buggy: log + accept (graceful degrade)."""

    skill = _make_skill(
        "buggy",
        "def verify(payload):\n    return xyz  # undefined name on purpose\n",
    )
    loader = _stub_skill_loader({"buggy": skill})

    fake = _install_fake_interpreter(
        monkeypatch,
        results=[
            _submit({"answer": 1}),
        ],
        verifier_results=[_verifier_name_error()],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(answer=1)\n```",
        ]
    )
    rlm = RLM.from_task(
        "Return 1.",
        outputs=["answer"],
        lm=lm,
        max_turns=4,
        timeout=5,
        skills=["buggy"],
        skill_loader=loader,
    )

    with caplog.at_level("WARNING", logger="fabric_rlm.runtime"):
        result = rlm.run()

    assert result.submitted is True
    assert result.payload == {"answer": 1}
    turn_types = [t.turn_type for t in result.trajectory]
    assert "verifier_repair" not in turn_types
    assert turn_types == ["normal"]
    # We did try to run the verifier, but it errored and was tolerated.
    assert len(fake.verifier_executed) == 1
    assert any(
        "non-AssertionError" in record.message and "buggy" in record.message
        for record in caplog.records
    )


def test_enable_verifier_false_skips_execution(monkeypatch) -> None:
    """``enable_verifier=False`` reverts to pre-C behaviour even with verifier-bearing skills."""

    skill = _make_skill(
        "mcm_stub",
        "def verify(payload):\n    assert False, 'always fails'\n",
    )
    loader = _stub_skill_loader({"mcm_stub": skill})

    fake = _install_fake_interpreter(
        monkeypatch,
        results=[
            _submit({"Q1": "x", "Q2": 0, "Q3": 1, "Q4": 1, "Q5": 0}),
        ],
    )
    lm = ScriptedLM(
        [
            "```python\nSUBMIT(Q1='x', Q2=0, Q3=1, Q4=1, Q5=0)\n```",
        ]
    )
    rlm = RLM.from_task(
        "Stub.",
        outputs=["Q1", "Q2", "Q3", "Q4", "Q5"],
        lm=lm,
        max_turns=4,
        timeout=5,
        skills=["mcm_stub"],
        skill_loader=loader,
        enable_verifier=False,
    )

    result = rlm.run()

    assert result.submitted is True
    assert fake.verifier_executed == []
