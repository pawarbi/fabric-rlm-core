"""The behaviour gate must not charge an environment failure to the pull request.

Written after a retired free-tier model slug 404'd on every question for a week.
Each 404 was recorded as a failing qid, and the gate reported five per-qid
"regressions" on every PR. Nothing had regressed, and a permanently red
informational check trains everyone to ignore it.

The rule under test: if a question never reached the model, its outcome is not
evidence about this PR, and the gate says so instead of counting it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from . import runner as runner_mod
from . import test_behavior_baseline as gate_mod


class _Res:
    """Minimal stand-in for a QuestionRun."""

    def __init__(self, qid: str, error_class: str | None, passed: bool = False):
        self.qid = qid
        self.error_class = error_class
        self.passed = passed
        self.reason = f"stub {error_class}"
        self.n_turns = 0
        self.elapsed_s = 0.0
        self.attempts = 1
        self.answer = None
        self.expected = None


@pytest.mark.parametrize("klass", ["auth", "runner_error"])
def test_gate_aborts_when_the_question_never_reached_the_model(klass, monkeypatch):
    """Both classes abort, and the message blames the environment, not the PR."""
    # A calibrated model, so the gate does not skip before reaching the stub.
    monkeypatch.setattr(gate_mod, "_require_api_key", lambda: None)

    def fake_run_question(q, model, **kw):
        return _Res(q.qid, klass)

    with patch.object(runner_mod, "run_question", fake_run_question):
        with pytest.raises(BaseException) as exc:
            gate_mod._run_gate("openai/gpt-4.1-mini", request=None)

    text = str(exc.value)
    assert "aborted" in text.lower()
    assert klass in text
    assert "NOT a model regression" in text
    # The specific failure mode this guards: it must not read as N regressions.
    assert "regressed" not in text


def test_a_wrong_answer_is_still_a_regression_signal(monkeypatch):
    """The gate must keep failing on real wrong answers.

    Without this, widening the abort set could silence the thing the gate is
    for. `wrong_answer` is deliberately outside _ABORT_CLASSES.
    """
    assert "wrong_answer" not in gate_mod._ABORT_CLASSES


def test_infra_is_not_an_abort_class():
    """Infra is retried once by the runner; a blip that survives that is a real
    failure worth seeing, not a reason to abandon the gate."""
    assert "infra" not in gate_mod._ABORT_CLASSES


def test_the_retired_slug_404_classifies_as_an_abort():
    """The exact error that ran red for a week, end to end through the classifier."""
    exc = RuntimeError(
        'NotFoundError: litellm.NotFoundError: OpenrouterException - '
        '{"error":{"message":"This model is unavailable for free. The paid '
        'version is available now - use this slug instead: openai/gpt-oss-120b",'
        '"code":404}}'
    )
    klass = runner_mod._classify_error(exc)
    assert klass in gate_mod._ABORT_CLASSES, (
        f"a retired-slug 404 classified as {klass!r}, which the gate would count "
        f"as a per-qid regression"
    )
