"""verified_task: blind double-solve with structural agreement and reconciliation."""
from __future__ import annotations

from types import MappingProxyType
from unittest.mock import patch

import pytest

import fabric_rlm.runtime as runtime_module
import fabric_rlm.verify as verify_module
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
    assert vr.selected_attempt_index == 0
    assert not vr.reconciliation_attempted
    assert not vr.reconciliation_succeeded
    assert not vr.fallback_used
    assert vr.reconciliation_reason is None


def test_disagreement_runs_reconciler_and_wins(monkeypatch):
    _wire(monkeypatch, [{"answer": "7"}, {"answer": "9"}, {"answer": "9"}])
    lm = ScriptedLM([CODE, CODE, CODE])
    vr = verified_task("How many?", outputs=["answer"], lm=lm, max_turns=3, timeout=5)
    assert vr.verdict == "reconciled"
    assert vr.result.payload["answer"] == "9"
    assert len(vr.attempts) == 3
    assert vr.answer_a == "7" and vr.answer_b == "9"
    assert vr.reconciliation_succeeded
    assert vr.selected_attempt_index == 2
    assert not vr.fallback_used
    assert vr.reconciliation_reason == "disagreement"


def test_empty_reconciler_falls_back_to_candidates(monkeypatch):
    # A and B disagree; the reconciler comes back empty. An empty answer trips
    # the runtime's output-validation retry, so solve C consumes two turns.
    _wire(monkeypatch, [{"answer": "7"}, {"answer": "9"},
                        {"answer": ""}, {"answer": ""}])
    lm = ScriptedLM([CODE, CODE, CODE, CODE])
    vr = verified_task("How many?", outputs=["answer"], lm=lm, max_turns=2, timeout=5)
    assert vr.verdict == "reconciled"
    assert vr.result.payload["answer"] == "7"  # falls back to candidate A
    assert vr.fallback_used
    assert vr.selected_attempt_index == 0
    assert vr.reconciliation_attempted
    assert not vr.reconciliation_succeeded


def test_agreement_never_calls_strong_reconciler(monkeypatch):
    _wire(monkeypatch, [{"answer": "42"}, {"answer": "42"}])
    lm = ScriptedLM([CODE, CODE])
    strong = ScriptedLM([])
    vr = verified_task(
        "How many?", outputs=["answer"], lm=lm, max_turns=2,
        reconcile_lm=strong, reconcile_max_turns=4,
    )
    assert vr.verdict == "agree"
    assert len(vr.attempts) == 2
    assert lm.calls == 2 and strong.calls == 0


def test_strong_reconciler_overrides_only_third_solve(monkeypatch):
    _wire(monkeypatch, [{"answer": "7"}, {"answer": "9"}, {"answer": "11"}])
    lm = ScriptedLM([CODE, CODE])
    strong = RecordingLM([CODE])
    inputs = {"data": [7, 9, 11]}
    outputs = {"answer": str}
    validated = []
    validated_context = []

    def validate(payload):
        validated.append(payload["answer"])

    def validate_context(payload, context):
        validated_context.append(payload["answer"])

    settings = dict(
        lm=lm, max_turns=2, timeout=5, output_validator=validate,
        output_validator_context=validate_context,
    )
    with patch.object(verify_module.RLM, "from_task", wraps=verify_module.RLM.from_task) as factory:
        vr = verified_task(
            "How many?", outputs=outputs, inputs=inputs,
            reconcile_lm=strong, reconcile_max_turns=4, **settings,
        )
    assert vr.verdict == "reconciled"
    assert vr.result is vr.attempts[2]
    assert vr.result.payload["answer"] == "11"
    assert lm.calls == 2 and strong.calls == 1
    assert validated == validated_context == ["7", "9", "11"]
    assert len(factory.call_args_list) == 3
    for index, call in enumerate(factory.call_args_list):
        expected = dict(settings, outputs=outputs, inputs=inputs)
        if index == 2:
            expected.update(lm=strong, max_turns=4)
        else:
            assert call.args == ("How many?",)
        assert call.kwargs == expected
        assert call.kwargs["inputs"] is inputs
        assert call.kwargs["outputs"] is outputs
    assert any("Analyst 1 answered: 7" in text and "Analyst 2 answered: 9" in text
               for text in strong.seen)


@pytest.mark.parametrize("base_turns,override", [(None, None), (2, None), (2, 1), (2, 3)])
@pytest.mark.parametrize("separate_lm", [False, True])
def test_reconciler_turn_budget_and_fallback(monkeypatch, base_turns, override, separate_lm):
    turns = override if override is not None else (base_turns if base_turns is not None else 20)
    _wire(monkeypatch, [{"answer": "7"}, {"answer": "9"}] + [{"answer": ""}] * turns)
    lm = ScriptedLM([CODE] * (2 if separate_lm else 2 + turns))
    strong = ScriptedLM([CODE] * turns) if separate_lm else None
    kwargs = {} if base_turns is None else {"max_turns": base_turns}
    vr = verified_task(
        "How many?", outputs=["answer"], lm=lm, reconcile_lm=strong,
        reconcile_max_turns=override, **kwargs,
    )
    assert vr.verdict == "reconciled"
    assert vr.result is vr.attempts[0]
    assert len(vr.attempts) == 3
    assert lm.calls == (2 if separate_lm else 2 + turns)
    if strong is not None:
        assert strong.calls == turns


@pytest.mark.parametrize("field_name,verdict", [(None, "agree"), ("other", "reconciled")])
def test_typed_mapping_selects_first_field_or_explicit_field(monkeypatch, field_name, verdict):
    _wire(monkeypatch, [
        {"count": 42, "other": "7"}, {"count": 42, "other": "9"},
        {"count": 42, "other": "11"},
    ])
    lm = ScriptedLM([CODE] * 3)
    vr = verified_task(
        "How many?", outputs=MappingProxyType({"count": int, "other": str}),
        field_name=field_name, lm=lm, max_turns=1,
    )
    assert vr.verdict == verdict
    assert vr.answer_a == ("42" if field_name is None else "7")
    assert vr.result.payload["count"] == 42
    assert lm.calls == (2 if field_name is None else 3)


@pytest.mark.parametrize("outputs", [[], {}, MappingProxyType({})])
def test_empty_outputs_fail_before_solving(monkeypatch, outputs):
    tasks = _stub_solves(monkeypatch, [])
    with pytest.raises(ValueError, match="outputs must contain at least one field name"):
        verified_task("How many?", outputs=outputs, field_name="answer")
    assert tasks == []


@pytest.mark.parametrize("outputs", [["answer"], {"answer": str}])
@pytest.mark.parametrize("field_name", ["missing", ""])
def test_unknown_field_fails_before_solving(monkeypatch, outputs, field_name):
    tasks = _stub_solves(monkeypatch, [])
    with pytest.raises(ValueError, match="not declared in outputs"):
        verified_task("Question", outputs=outputs, field_name=field_name)
    assert tasks == []


def test_typed_output_default_still_repairs_wrong_type(monkeypatch):
    _wire(monkeypatch, [{"answer": 42}, {"answer": "42"}, {"answer": "42"}])
    lm = ScriptedLM([CODE] * 3)
    vr = verified_task("Answer as text.", outputs={"answer": str}, lm=lm, max_turns=2)
    assert vr.verdict == "agree"
    assert vr.result.payload == {"answer": "42"}
    assert lm.calls == 3


@pytest.mark.parametrize("turns", [0, -1, 1.5, "2", True, False])
def test_invalid_reconcile_turn_budget_fails_before_solving(monkeypatch, turns):
    tasks = _stub_solves(monkeypatch, [])
    with pytest.raises(ValueError, match="reconcile_max_turns must be a positive integer"):
        verified_task("How many?", outputs=["answer"], reconcile_max_turns=turns)
    assert tasks == []


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


def test_agree_sign_is_part_of_the_number():
    # Normalization strips punctuation, and the minus sign with it, so the
    # signed numeric comparison must run before the normalized-equality
    # shortcut or a sign flip is waved through as agreement.
    for a, b in [("10", "-10"), ("10%", "-10%"), ("$5", "-$5"), ("+10", "-10"),
                 ("Growth was 10", "Growth was -10"), ("-10; A", "10; A"),
                 ("(10)", "10"), ("($5)", "5")]:
        assert not answers_agree(a, b), (a, b)
        assert not answers_agree(b, a), (b, a)
    assert answers_agree("Growth was -3.5%", "-3.5% growth")
    assert answers_agree("$-5", "-$5")


def test_agree_accounting_negative_and_range_hyphen():
    assert answers_agree("(10)", "-10")            # accounting negative
    assert answers_agree("($5)", "-5")
    assert answers_agree("(10 items)", "10 items")  # parentheses around prose, not a sign
    assert answers_agree("2024-2025", "2024 to 2025")  # hyphen is a range separator
    assert not answers_agree("FY2024 revenue 10", "FY2025 revenue 10")


def test_agree_blank_answers_never_agree():
    assert not answers_agree("", "")
    assert not answers_agree("  ", "")
    assert not answers_agree(None, None)
    assert not answers_agree("", "42")


def test_agree_containment_is_whole_word():
    assert not answers_agree("Mark", "Denmark")
    assert not answers_agree("Ann", "Annie")
    assert answers_agree("Denmark", "the answer is Denmark")


def test_sign_flip_reconciles(monkeypatch):
    _wire(monkeypatch, [{"answer": "10"}, {"answer": "-10"}, {"answer": "-10"}])
    lm = ScriptedLM([CODE, CODE, CODE])
    vr = verified_task("Net change?", outputs=["answer"], lm=lm, max_turns=3, timeout=5)
    assert vr.verdict == "reconciled"
    assert vr.result.payload["answer"] == "-10"
    assert len(vr.attempts) == 3
    assert vr.reconciliation_attempted
    assert vr.reconciliation_succeeded
    assert not vr.fallback_used
    assert vr.selected_attempt_index == 2
    assert vr.reconciliation_reason == "disagreement"


class RecordingLM(ScriptedLM):
    def __init__(self, responses):
        super().__init__(responses)
        self.seen: list[str] = []

    def __call__(self, *, messages):
        self.seen.extend(str(m.get("content", "")) for m in messages)
        return super().__call__(messages=messages)


def test_two_blank_solves_reconcile(monkeypatch):
    # Blank answers trip the runtime's validation retry, so each blank solve
    # consumes two turns. Two blanks are not agreement: the reconciler runs and
    # sees that neither analyst answered.
    _wire(monkeypatch, [{"answer": ""}, {"answer": ""},
                        {"answer": ""}, {"answer": ""},
                        {"answer": "5"}])
    lm = RecordingLM([CODE] * 5)
    vr = verified_task("How many?", outputs=["answer"], lm=lm, max_turns=2, timeout=5)
    assert vr.verdict == "reconciled"
    assert vr.result.payload["answer"] == "5"
    assert len(vr.attempts) == 3
    assert any("Analyst 1 answered: (no answer)" in text for text in lm.seen)


# ------------------------------------------------- sign variants and lists --
def test_agree_unicode_minus_and_dashes_are_signs():
    assert not answers_agree("−10", "10")          # U+2212 MINUS SIGN
    assert not answers_agree("–10%", "10%")        # en dash used as a minus
    assert answers_agree("−10", "-10")
    assert answers_agree("2024–2025", "2024 to 2025")  # en dash range, not a sign


def test_agree_comma_lists_need_identical_item_sets():
    assert not answers_agree("A, B", "A, B, C")          # containment must not hide a missing member
    assert not answers_agree("A, B, C", "A, B")
    assert answers_agree("A, B, and C", "C; B; A")       # Oxford "and" is not an item
    assert answers_agree("$1,077", "1,077 dollars")      # thousands separator is not a list
    assert answers_agree("Denmark", "the answer is Denmark")


# ------------------------------------------------------- attempt validity --
class _Stub:
    def __init__(self, answer, *, submitted=True, failure_reason=None, integrity_ok=True):
        self.submitted = submitted
        self.payload = {"answer": answer} if submitted else None
        self.failure_reason = failure_reason
        self.integrity_ok = integrity_ok
        self.total_prompt_tokens = 1
        self.total_completion_tokens = 1


def _stub_solves(monkeypatch, results):
    import fabric_rlm.verify as verify_module

    queue = list(results)
    tasks = []

    class FakeRLM:
        @staticmethod
        def from_task(text, **kwargs):
            tasks.append(text)

            class Runner:
                @staticmethod
                def run():
                    return queue.pop(0)

            return Runner()

    monkeypatch.setattr(verify_module, "RLM", FakeRLM)
    return tasks


def test_all_failed_solves_are_a_failed_verdict(monkeypatch):
    failed = lambda: _Stub("", submitted=False, failure_reason="max_turns")
    _stub_solves(monkeypatch, [failed(), failed(), failed()])
    vr = verified_task("How many?", outputs=["answer"])
    assert vr.verdict == "failed" and not vr.ok
    assert vr.result.submitted is False and vr.result.failure_reason == "max_turns"
    assert len(vr.attempts) == 3


def test_failed_solve_is_never_a_candidate_even_with_custom_agree(monkeypatch):
    tasks = _stub_solves(monkeypatch, [
        _Stub("", submitted=False, failure_reason="worker_timeout"),
        _Stub("42"),
        _Stub("42"),
    ])
    vr = verified_task("How many?", outputs=["answer"], agree=lambda a, b: True)
    assert vr.verdict == "reconciled" and vr.result.payload["answer"] == "42"
    assert "Analyst 1 answered: (no answer)" in tasks[-1]


def test_agreement_prefers_the_integrity_clean_candidate(monkeypatch):
    unresolved, clean = _Stub("42", integrity_ok=False), _Stub("42", integrity_ok=True)
    _stub_solves(monkeypatch, [unresolved, clean])
    assert verified_task("How many?", outputs=["answer"]).result is clean
    _stub_solves(monkeypatch, [clean, unresolved])
    assert verified_task("How many?", outputs=["answer"]).result is clean
    both = [_Stub("42"), _Stub("42")]
    _stub_solves(monkeypatch, both)
    assert verified_task("How many?", outputs=["answer"]).result is both[0]


def test_reconciler_failure_falls_back_to_a_usable_candidate(monkeypatch):
    usable = _Stub("9")
    _stub_solves(monkeypatch, [
        _Stub("", submitted=False, failure_reason="max_turns"),
        usable,
        _Stub("", submitted=False, failure_reason="max_turns"),
    ])
    vr = verified_task("How many?", outputs=["answer"])
    assert vr.verdict == "reconciled" and vr.result is usable
    assert vr.fallback_used
    assert vr.selected_attempt_index == 1
    assert not vr.reconciliation_succeeded
    assert vr.reconciliation_reason == "unavailable_answer"
