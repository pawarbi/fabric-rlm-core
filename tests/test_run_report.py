"""RLMResult.report(): deterministic facts about what a run did.

The cases here are the ones that were expensive to reconstruct by hand from raw
turns, which is why the report exists at all:

* a run where no code ran is not a wrong answer, and must not read as one
* a run that hit the turn ceiling is truncated, not necessarily wrong
* cached prompt share matters, because a naive token count once overstated a
  feature's cost by 3x and reversed a conclusion
"""

from __future__ import annotations

import pytest

from fabric_rlm import RLM


def code(body: str) -> str:
    return f"```python\n{body}\n```"


class ScriptedLM:
    def __init__(self, turns):
        self.turns, self.i = list(turns), 0

    def __call__(self, messages=None, prompt=None, **kw):
        if self.i >= len(self.turns):
            return ["# nothing left"]
        t = self.turns[self.i]
        self.i += 1
        return [t]


class ProseOnlyLM:
    """Answers, but never emits a code block."""

    def __call__(self, messages=None, prompt=None, **kw):
        return ["I will not write code."]


def run(turns, **kw):
    kw.setdefault("max_turns", 6)
    kw.setdefault("timeout", 60)
    return RLM.task(task="t", outputs=["answer"], lm=ScriptedLM(turns), **kw).run()


# --- ran_any_code ------------------------------------------------------------

def test_prose_only_model_ran_no_code():
    """The distinction the property exists for.

    A model that replies in prose was still *reached* - it answered. Nothing
    executed, so the run is not evidence the model got the task wrong. This is
    why the property is not called "reached_model": an LM error raises out of
    run() rather than returning a result at all.
    """
    r = RLM.task(task="t", outputs=["answer"], lm=ProseOnlyLM(),
                 max_turns=2, timeout=60).run()
    assert r.ran_any_code is False
    assert r.n_turns == 0
    text = r.report()
    assert "NO CODE RAN" in text
    assert "not a wrong answer" in text


def test_normal_run_ran_code():
    r = run([code("x = 1"), code("SUBMIT(answer='ok')")])
    assert r.ran_any_code is True
    assert r.submitted is True


# --- the three headline shapes ----------------------------------------------

def test_success_names_the_submit_turn_and_outputs():
    r = run([code("x = 1"), code("SUBMIT(answer='ok')")])
    text = r.report()
    assert "SUBMITTED on turn 2" in text
    assert "answer" in text


def test_ceiling_is_called_out_as_truncation_not_failure():
    r = run([code("x = 1"), code("y = 2")], max_turns=2)
    text = r.report()
    assert "NOT SUBMITTED" in text
    assert "Ran out of turns" in text
    assert r.report(as_dict=True)["hit_ceiling"] is True


def test_submitted_run_never_reports_hit_ceiling():
    """A run that submitted on its last turn finished; it did not run out."""
    r = run([code("x = 1"), code("SUBMIT(answer='ok')")], max_turns=2)
    assert r.submitted is True
    assert r.report(as_dict=True)["hit_ceiling"] is False


# --- errors ------------------------------------------------------------------

def test_errors_are_counted_and_classified():
    r = run([code("raise ValueError(9)"), code("SUBMIT(answer='ok')")])
    facts = r.report(as_dict=True)
    assert facts["errors"] == 1
    assert facts["error_kinds"].get("ValueError") == 1
    assert "ValueError" in r.report()


def test_a_run_with_errors_can_still_have_submitted():
    """Errors are not failures; the loop is expected to recover from them."""
    r = run([code("raise ValueError(9)"), code("SUBMIT(answer='ok')")])
    assert r.submitted is True
    assert r.report(as_dict=True)["errors"] == 1


# --- shape and safety --------------------------------------------------------

def test_dict_form_is_json_serialisable():
    import json

    r = run([code("x = 1"), code("SUBMIT(answer='ok')")])
    json.dumps(r.report(as_dict=True))     # must not raise


def test_report_is_offline_and_consults_no_model():
    """The LM is exhausted before report() is called; it must not be invoked."""
    lm = ScriptedLM([code("SUBMIT(answer='ok')")])
    r = RLM.task(task="t", outputs=["answer"], lm=lm, max_turns=3, timeout=60).run()
    before = lm.i
    r.report()
    r.report(as_dict=True)
    assert lm.i == before, "report() called the LM"


def test_report_never_raises_on_a_bare_result():
    """Report is a diagnostic; it must not fail when fields are missing."""
    from fabric_rlm.runtime import RLMResult
    from fabric_rlm.trajectory import Trajectory

    bare = RLMResult(submitted=False, payload=None, trajectory=Trajectory(turns=[]),
                     final_state={})
    text = bare.report()
    assert isinstance(text, str) and text
    facts = bare.report(as_dict=True)
    assert facts["turns"] == 0
    assert facts["ran_any_code"] is False


@pytest.mark.parametrize("n_turns", [1, 3])
def test_turn_count_matches_the_trajectory(n_turns):
    r = run([code(f"x = {i}") for i in range(n_turns)], max_turns=n_turns)
    assert r.report(as_dict=True)["turns"] == r.n_turns == n_turns
