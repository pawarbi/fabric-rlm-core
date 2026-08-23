"""Slice 3 tests: verifier wrapper around the v7-dspy engine.

Validates that:
  - When ``output_validator`` accepts the payload, the loop ships immediately.
  - When ``output_validator`` rejects, we re-call dspy.RLM with feedback
    prepended to the first string input. Bounded retry (≤2).
  - When all retries exhausted, the final RLMResult has ``submitted=False``
    and a ``failure_reason``.
"""

from __future__ import annotations

import dspy
import pytest

from fabric_rlm import RLM, RLMResult


class _ScriptedLM(dspy.LM):
    """Returns a sequence of canned responses, one per call."""

    def __init__(self, scripted_codes: list[str]) -> None:
        super().__init__(model="scripted", model_type="chat")
        self._codes = list(scripted_codes)
        self.calls = 0
        self.requests: list[str] = []

    def __call__(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        self.requests.append(str(messages if messages is not None else prompt))
        if self._codes:
            code = self._codes.pop(0)
        else:
            code = "SUBMIT(answer=999)"
        self.calls += 1
        return [
            "[[ ## reasoning ## ]]\nThinking.\n\n"
            f"[[ ## code ## ]]\n```python\n{code}\n```\n\n"
            "[[ ## completed ## ]]\n"
        ]


def _validator_must_be_42(payload: dict) -> None:
    assert payload.get("answer") == 42, "answer must be 42"


def test_output_validator_accepts_first_attempt() -> None:
    lm = _ScriptedLM(["SUBMIT(answer=42)"])
    rlm = RLM(
        signature="question -> answer: int",
        lm=lm,
        engine="v7-dspy",
        output_validator=_validator_must_be_42,
    )
    result = rlm(question="x")
    assert result.submitted is True
    assert result.payload["answer"] == 42


def test_output_validator_rejects_then_succeeds_on_retry() -> None:
    """First SUBMIT(7) gets rejected; second SUBMIT(42) is accepted."""
    lm = _ScriptedLM(["SUBMIT(answer=7)", "SUBMIT(answer=42)"])
    rlm = RLM(
        signature="question -> answer: int",
        lm=lm,
        engine="v7-dspy",
        output_validator=_validator_must_be_42,
    )
    result = rlm(question="x")

    assert result.submitted is True, f"expected retry to succeed; reason={result.failure_reason}"
    assert result.payload["answer"] == 42
    history = result.trajectory.metadata.get("verifier_repair_history", [])
    assert len(history) >= 1, "expected at least one verifier_repair history entry"


def test_typed_output_mapping_retries_on_v7_engine() -> None:
    lm = _ScriptedLM(
        [
            "SUBMIT(result='South')",
            "SUBMIT(result={'top_region': 'South'})",
        ]
    )
    rlm = RLM.task(
        "Return a structured result.",
        inputs={"question": "x"},
        outputs={"result": dict},
        lm=lm,
        engine="dspy",
    )

    result = rlm.run()

    assert result.submitted is True
    assert result.payload == {"result": {"top_region": "South"}}
    assert lm.calls >= 2


def test_typed_output_mapping_prevents_dspy_bool_to_int_coercion() -> None:
    lm = _ScriptedLM(["SUBMIT(result=True)", "SUBMIT(result=7)"])
    result = RLM.task(
        "Return an integer.",
        inputs={"question": "x"},
        outputs={"result": int},
        lm=lm,
        engine="dspy",
    ).run()

    assert result.submitted is True
    assert result.payload == {"result": 7}
    assert lm.calls >= 2


def test_typed_output_feedback_reaches_dspy_with_numeric_only_inputs() -> None:
    lm = _ScriptedLM(["SUBMIT(result='7')", "SUBMIT(result=7)"])
    result = RLM.task(
        "Return an integer.",
        inputs={"n": 1},
        outputs={"result": int},
        lm=lm,
        engine="dspy",
    ).run()

    assert result.submitted is True
    assert result.payload == {"result": 7}
    assert any("must be int, got str" in request for request in lm.requests[1:])


def test_output_validator_exhausts_retries() -> None:
    """All attempts fail; result is unsubmitted with failure_reason set."""
    lm = _ScriptedLM([
        "SUBMIT(answer=1)",
        "SUBMIT(answer=2)",
        "SUBMIT(answer=3)",
        "SUBMIT(answer=4)",
    ])
    rlm = RLM(
        signature="question -> answer: int",
        lm=lm,
        engine="v7-dspy",
        output_validator=_validator_must_be_42,
    )
    result = rlm(question="x")
    assert result.submitted is False
    assert result.failure_reason is not None
    assert "verifier" in result.failure_reason.lower() or "rejected" in result.failure_reason.lower()
    history = result.trajectory.metadata.get("verifier_repair_history", [])
    assert len(history) >= 1


def test_skill_verifier_actually_runs_in_v7_path(tmp_path) -> None:
    """REGRESSION: skill verifier must execute (catches unstarted-Interpreter bug).

    A skill with a verifier that asserts ``answer == 42`` is loaded explicitly.
    First SUBMIT(7) must be REJECTED by the skill verifier (not silently
    accepted via 'graceful degrade'); second SUBMIT(42) must be accepted.
    """
    import textwrap
    from fabric_rlm.skill_loader import SkillLoader

    skill_md = textwrap.dedent(
        """\
        ---
        applies_when:
          keywords: [foo]
        specificity: domain
        ---

        # Foo Skill

        Body.

        ## Required verifier

        ```python
        def verify(payload):
            assert payload.get("answer") == 42, "answer must be 42"
        ```
        """
    )
    (tmp_path / "foo.md").write_text(skill_md, encoding="utf-8")
    loader = SkillLoader(skill_dir=tmp_path)
    # Sanity: skill must have a verifier source.
    assert loader.load("foo").verifier_source, "verifier_source must be parsed from skill"

    lm = _ScriptedLM(["SUBMIT(answer=7)", "SUBMIT(answer=42)"])
    rlm = RLM(
        signature="question -> answer: int",
        lm=lm,
        engine="v7-dspy",
        enable_router=False,
        skills=["foo"],
        skill_loader=loader,
    )
    result = rlm(question="anything foo")

    assert result.submitted is True, (
        f"expected verifier-driven retry to succeed; reason={result.failure_reason}"
    )
    assert result.payload["answer"] == 42
    history = result.trajectory.metadata.get("verifier_repair_history", [])
    assert any(h.get("skill") == "foo" for h in history), (
        f"expected skill verifier to fire (rejecting answer=7); history={history}"
    )


def test_verifier_feedback_prepended_to_first_string_input() -> None:
    """The retry's input must contain the rejection feedback prepended."""
    captured_inputs: list[str] = []

    class CapturingLM(dspy.LM):
        def __init__(self) -> None:
            super().__init__(model="cap", model_type="chat")
            self.calls = 0

        def __call__(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
            self.calls += 1
            joined = ""
            if messages:
                joined = "\n".join(
                    m.get("content", "") for m in messages if isinstance(m, dict)
                )
            captured_inputs.append(joined)
            answer = 42 if self.calls >= 2 else 7
            return [
                "[[ ## reasoning ## ]]\nGo.\n\n"
                f"[[ ## code ## ]]\n```python\nSUBMIT(answer={answer})\n```\n\n"
                "[[ ## completed ## ]]\n"
            ]

    lm = CapturingLM()
    rlm = RLM(
        signature="question -> answer: int",
        lm=lm,
        engine="v7-dspy",
        output_validator=_validator_must_be_42,
    )
    result = rlm(question="initial question text")
    assert result.submitted is True
    assert len(captured_inputs) >= 2, "expected at least 2 LM calls (initial + retry)"
    # The 2nd action prompt should contain the verifier feedback.
    assert "VERIFIER FEEDBACK" in captured_inputs[1] or "rejected" in captured_inputs[1].lower()
