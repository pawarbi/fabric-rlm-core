"""verified_task: blind double-solve with structural agreement and reconciliation."""
from __future__ import annotations

import fabric_rlm.runtime as runtime_module
from fabric_rlm.verify import VerifiedResult, answers_agree, verified_task


class _LMResponse:
    def __init__(self, content: str, usage: dict | None = None):
        self.content = content
        if usage is not None:
            self.usage = usage


class ScriptedLM:
    """One response per RLM turn, shared across sequential solves."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, *, messages):
        assert self.responses, "No scripted responses left"
        self.calls += 1
        text = self.responses.pop(0)
        return _LMResponse(text, {"prompt_tokens": 100, "completion_tokens": 10})


class _ExecResult:
    def __init__(self, payload):
        self.stdout = "ok"
        self.stderr = ""
        self.error = None
        self.submitted = True
        self.submit_payload = payload
        self.state: dict = {}
        self.ok = True


class SequencedInterpreter:
    """Pops one submit payload per executed turn, across solves."""

    def __init__(self, payloads):
        self.payloads = list(payloads)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def configure_lm(self, _spec):
        pass

    def set_inputs(self, _inputs):
        pass

    def execute(self, _code):
        assert self.payloads, "No scripted payloads left"
        return _ExecResult(self.payloads.pop(0))


CODE = "```python\nSUBMIT(answer=x)\n```"


def _wire(monkeypatch, payloads):
    fake = SequencedInterpreter(payloads)
    monkeypatch.setattr(runtime_module, "Interpreter", lambda **kwargs: fake)
    return fake


# ---------------------------------------------------------------- agreement --
def test_agreement_stops_at_two_solves(monkeypatch):
    _wire(monkeypatch, [{"answer": "42.0"}, {"answer": "42.0"}])
    lm = ScriptedLM([CODE, CODE])
    vr = verified_task("How many?", outputs=["answer"], lm=lm, max_turns=3, timeout=5)
    assert isinstance(vr, VerifiedResult)
    assert vr.verdict == "agree"
    assert vr.result.payload["answer"] == "42.0"
    assert len(vr.attempts) == 2
    assert lm.calls == 2                       # no reconciler consumed
    assert vr.total_prompt_tokens == 200       # billed across the ensemble


def test_disagreement_runs_reconciler_and_wins(monkeypatch):
    _wire(monkeypatch, [{"answer": "7"}, {"answer": "9"}, {"answer": "9"}])
    lm = ScriptedLM([CODE, CODE, CODE])
    vr = verified_task("How many?", outputs=["answer"], lm=lm, max_turns=3, timeout=5)
    assert vr.verdict == "reconciled"
    assert vr.result.payload["answer"] == "9"
    assert len(vr.attempts) == 3
    assert vr.answer_a == "7" and vr.answer_b == "9"


def test_empty_reconciler_falls_back_to_candidates(monkeypatch):
    # A and B disagree; the reconciler comes back empty. An empty answer trips
    # the runtime's output-validation retry, so solve C consumes two turns.
    _wire(monkeypatch, [{"answer": "7"}, {"answer": "9"},
                        {"answer": ""}, {"answer": ""}])
    lm = ScriptedLM([CODE, CODE, CODE, CODE])
    vr = verified_task("How many?", outputs=["answer"], lm=lm, max_turns=2, timeout=5)
    assert vr.verdict == "reconciled"
    assert vr.result.payload["answer"] == "7"  # falls back to candidate A


# ---------------------------------------------------------- agreement logic --
def test_agree_numbers_must_match_exactly():
    assert not answers_agree("1077", "1080")
    assert not answers_agree("1077", "1077.0")     # format difference -> reconcile
    assert answers_agree("Total: 42.5", "the total is 42.5")


def test_agree_lists_need_identical_item_sets():
    assert not answers_agree("A; B; C", "A; B; C; D")   # missing member = disagree
    assert answers_agree("B; A; C", "A; B; C")          # order-insensitive


def test_agree_short_vs_verbose_phrasing():
    assert answers_agree("Negotiation", "Negotiation stage")
    assert not answers_agree("MI", "TX")
