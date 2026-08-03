"""A worker timeout should be recoverable, not fatal.

Killing the worker loses the namespace, so recovery means restarting it,
re-binding inputs, and telling the model its variables are gone. Ending the run
instead discards every prior turn: on a 246-task benchmark that cost 20 tasks,
several of which had finished the analysis and were writing output.
"""

from __future__ import annotations

import pytest

from fabric_rlm import RLM


class ScriptedLM:
    """Returns each scripted turn in order; records what it was told."""

    def __init__(self, turns: list[str]):
        self.turns = list(turns)
        self.i = 0
        self.seen: list[str] = []

    def __call__(self, messages=None, prompt=None, **kwargs):
        if messages:
            self.seen.append(messages[-1].get("content", ""))
        if self.i >= len(self.turns):
            return ["SUBMIT(answer='ran out of scripted turns')"]
        turn = self.turns[self.i]
        self.i += 1
        return [turn]


SLOW = "```python\nimport time\ntime.sleep(30)\n```"
FAST = "```python\nSUBMIT(answer='recovered')\n```"


def test_one_recovery_by_default():
    """The default allows a single recovery.

    A timeout is rare (23 of 983 task runs across four full benchmark runs)
    but costs the whole run when fatal. One recovery buys the "tell the model
    it was too slow and let it adapt" case; the ceiling on a doomed run is one
    extra timeout period rather than several.
    """
    lm = ScriptedLM([SLOW, FAST])
    result = RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=4,
                      timeout=2).run()
    assert result.submitted is True
    assert result.payload["answer"] == "recovered"


def test_opting_out_restores_fail_fast():
    """recover_worker_timeouts=0 is the old behaviour, still available."""
    lm = ScriptedLM([SLOW, FAST])
    result = RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=4,
                      timeout=2, recover_worker_timeouts=0).run()
    assert result.submitted is False
    assert result.failure_reason == "worker_timeout"


def test_default_still_bounded():
    """Two timeouts exceed the default budget of one, so the run ends."""
    lm = ScriptedLM([SLOW, SLOW, FAST])
    result = RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=6,
                      timeout=2).run()
    assert result.submitted is False
    assert result.failure_reason == "worker_timeout"


def test_recovery_lets_the_run_continue():
    lm = ScriptedLM([SLOW, FAST])
    result = RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=4,
                      timeout=2, recover_worker_timeouts=1).run()
    assert result.submitted is True, "run should survive one timeout"
    assert result.payload["answer"] == "recovered"


def test_model_is_told_the_namespace_is_gone():
    lm = ScriptedLM([SLOW, FAST])
    RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=4, timeout=2,
             recover_worker_timeouts=1).run()
    nudge = "\n".join(lm.seen).lower()
    assert "restarted" in nudge
    assert "variable" in nudge, "must warn that state was lost"


def test_budget_is_finite():
    """Two timeouts with a budget of one still ends the run."""
    lm = ScriptedLM([SLOW, SLOW, FAST])
    result = RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=6,
                      timeout=2, recover_worker_timeouts=1).run()
    assert result.submitted is False
    assert result.failure_reason == "worker_timeout"


def test_inputs_survive_the_restart():
    """Bound inputs must be usable again after the worker is replaced."""
    lm = ScriptedLM([
        SLOW,
        "```python\nSUBMIT(answer=str(value * 2))\n```",
    ])
    result = RLM.task(task="t", inputs={"value": 21}, outputs=["answer"],
                      lm=lm, max_turns=4, timeout=2,
                      recover_worker_timeouts=1).run()
    assert result.submitted is True
    assert result.payload["answer"] == "42", "input was not re-bound"


@pytest.mark.parametrize("budget", [0, 1, 3])
def test_budget_is_accepted_and_clamped(budget):
    rlm = RLM.task(task="t", outputs=["a"], lm=ScriptedLM([FAST]),
                   recover_worker_timeouts=budget)
    assert rlm.recover_worker_timeouts == budget
