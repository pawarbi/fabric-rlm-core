"""BUG-LIB-3: final-turn give-up prompt.

When the v6 run loop is one turn from exhausting its budget, the user-message
feedback should explicitly tell the LM that this is its FINAL turn and that it
must submit its best current answer (rather than requesting more inspection or
running another exploratory step that will never get a chance to land).

Trace mining over 1,354 SSB turns identified 12 traces that burned the full
budget without ever calling SUBMIT() — pure ``failure_reason="max_turns"``
exits with no answer payload. A clear last-turn nudge in the feedback message
recovers many of these by signaling that "good enough now" beats "perfect
later that never happens".

The fix is library-general: it applies to any iterative RLM workload, not the
SSB benchmark specifically.
"""

from __future__ import annotations

from fabric_rlm import RLM
from fabric_rlm.interpreter import ExecResult


def _exec(ok=True, stdout="", stderr="", error=None, state=None, submit_payload=None):
    return ExecResult(
        ok=ok, submitted=submit_payload is not None,
        stdout=stdout, stderr=stderr,
        state=state or {}, error=error, submit_payload=submit_payload,
    )


class ScriptedLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def __call__(self, *, messages):
        self.messages.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("No scripted responses left")
        return self.responses.pop(0)


def _wrap_code(body: str) -> str:
    return "```python\n" + body + "\n```"


def _last_user_msg(messages):
    return next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "",
    )


# ---------------------------------------------------------------------------
# Direct unit tests on _format_feedback
# ---------------------------------------------------------------------------


class TestFinalTurnNudgeContent:
    def _rlm(self, max_turns=5):
        lm = ScriptedLM([])
        return RLM.from_task("Return answer.", outputs=["answer"], lm=lm,
                             max_turns=max_turns, timeout=5)

    def test_non_final_turn_uses_default_close(self):
        rlm = self._rlm(max_turns=5)
        result = _exec(ok=False, error="NameError: foo", state={"x": 1})
        feedback = rlm._format_feedback(result, turn=1, is_final_turn=False)
        assert "Continue with one complete Python code block" in feedback
        assert "FINAL TURN" not in feedback
        assert "last turn" not in feedback.lower()

    def test_final_turn_emits_explicit_give_up_nudge(self):
        rlm = self._rlm(max_turns=5)
        result = _exec(ok=False, error="NameError: foo", state={"x": 1})
        feedback = rlm._format_feedback(result, turn=4, is_final_turn=True)
        assert "FINAL TURN" in feedback
        assert "SUBMIT" in feedback
        assert "Continue with one complete Python code block" not in feedback

    def test_final_turn_nudge_includes_state_keys(self):
        """Even on the final turn the LM should still see what state is
        available so it can submit something derived from prior work."""
        rlm = self._rlm(max_turns=3)
        result = _exec(ok=True, stdout="hello", state={"sales": 42, "tax": 3})
        feedback = rlm._format_feedback(result, turn=2, is_final_turn=True)
        assert "sales" in feedback
        assert "tax" in feedback

    def test_final_turn_default_kwarg_is_false(self):
        """Backward compat: callers that don't pass is_final_turn get the old
        non-final behavior."""
        rlm = self._rlm(max_turns=5)
        result = _exec(ok=True, stdout="ok")
        feedback = rlm._format_feedback(result, turn=1)
        assert "FINAL TURN" not in feedback
        assert "Continue with one complete Python code block" in feedback


# ---------------------------------------------------------------------------
# End-to-end behaviour: nudge actually flips submitting on the final turn
# ---------------------------------------------------------------------------


class TestFinalTurnDetection:
    def test_lm_that_only_submits_when_nudged_does_submit(self):
        """LM does not submit on its own; once it sees the FINAL TURN nudge in
        the user message it submits."""

        class NudgeAwareLM:
            def __init__(self):
                self.messages = []
                self.scripted = [
                    _wrap_code("print('inspecting')"),
                    _wrap_code("print('still inspecting')"),
                    _wrap_code("print('almost there')"),
                ]

            def __call__(self, *, messages):
                self.messages.append([dict(m) for m in messages])
                if self.scripted:
                    return self.scripted.pop(0)
                if "FINAL TURN" in _last_user_msg(messages):
                    return _wrap_code("SUBMIT(answer='best-effort')")
                return _wrap_code("print('still going')")

        lm = NudgeAwareLM()
        rlm = RLM.from_task("Compute and submit answer", outputs=["answer"],
                            lm=lm, max_turns=4, timeout=5)
        result = rlm.run()

        assert result.submitted, (
            f"LM was supposed to submit on the final turn after seeing the "
            f"FINAL TURN nudge. failure_reason={result.failure_reason}"
        )
        assert result.payload == {"answer": "best-effort"}

    def test_lm_that_ignores_nudge_still_hits_max_turns(self):
        """The nudge encourages submit but doesn't force it. A model that
        ignores the nudge entirely still ends at max_turns without crashing."""
        responses = [_wrap_code("print('ignoring')") for _ in range(5)]
        lm = ScriptedLM(responses)
        rlm = RLM.from_task("Whatever", outputs=["answer"], lm=lm,
                            max_turns=3, timeout=5)
        result = rlm.run()
        assert not result.submitted
        assert result.failure_reason == "max_turns"

    def test_nudge_only_on_final_iteration_not_earlier(self):
        """Nudge appears ONCE, exclusively on the final LM call."""
        responses = [
            _wrap_code("print('1')"),
            _wrap_code("print('2')"),
            _wrap_code("print('3')"),
            _wrap_code("print('4')"),
        ]
        lm = ScriptedLM(responses)
        rlm = RLM.from_task("Whatever", outputs=["answer"], lm=lm,
                            max_turns=4, timeout=5)
        rlm.run()

        nudge_seen_at = [
            idx for idx, msgs in enumerate(lm.messages)
            if "FINAL TURN" in _last_user_msg(msgs)
        ]
        # max_turns=4 → 4 LM calls (indices 0..3). Nudge in call index 3 only.
        assert nudge_seen_at == [3], (
            f"FINAL TURN nudge should appear only on the last LM call, "
            f"but appeared at call indices {nudge_seen_at}"
        )

    def test_max_turns_one_no_crash(self):
        """Edge case: max_turns=1 → only one LM call, no feedback messages."""
        responses = [_wrap_code("print('only turn')")]
        lm = ScriptedLM(responses)
        rlm = RLM.from_task("Whatever", outputs=["answer"], lm=lm,
                            max_turns=1, timeout=5)
        result = rlm.run()
        assert result.failure_reason == "max_turns"
        assert len(lm.messages) == 1

    def test_truncated_response_on_penultimate_turn_includes_final_nudge(self):
        """If the LM emits a truncated response on the penultimate turn, the
        truncation-retry user message that the LM sees on the FINAL turn must
        carry the FINAL TURN suffix.

        max_turns=2: turn 1 emits truncated, retry is turn 2 (the final).
        """
        responses = [
            "```python\nprint('truncated",  # 1 backtick triple -> _looks_truncated True
            _wrap_code("SUBMIT(answer='ok')"),
        ]
        lm = ScriptedLM(responses)
        rlm = RLM.from_task("Whatever", outputs=["answer"], lm=lm,
                            max_turns=2, timeout=5)
        rlm.run()

        # LM call 0 = first turn (no nudge possible there).
        # LM call 1 = second/final turn; user message MUST contain FINAL TURN.
        assert len(lm.messages) >= 2
        last_user = _last_user_msg(lm.messages[1])
        assert "truncated before the closing code fence" in last_user, (
            "Sanity: ensure truncation path was taken"
        )
        assert "FINAL TURN" in last_user, (
            "Truncated-response retry that lands on the final turn must "
            "include the FINAL TURN nudge."
        )

    def test_truncated_response_not_yet_final_omits_nudge(self):
        """Truncation retry that is NOT the final turn must not include the
        nudge (avoid premature urgency)."""
        responses = [
            "```python\nprint('truncated",
            _wrap_code("print('ok')"),
            _wrap_code("SUBMIT(answer='ok')"),
        ]
        lm = ScriptedLM(responses)
        rlm = RLM.from_task("Whatever", outputs=["answer"], lm=lm,
                            max_turns=3, timeout=5)
        rlm.run()
        # Turn 2 is the truncation retry; turn 3 is the final.
        # The truncation-retry user message at call index 1 must NOT have nudge.
        assert "truncated before the closing code fence" in _last_user_msg(lm.messages[1])
        assert "FINAL TURN" not in _last_user_msg(lm.messages[1])


# ---------------------------------------------------------------------------
# Non-regression: existing happy-path runs unchanged
# ---------------------------------------------------------------------------


class TestNonRegression:
    def test_two_turn_run_normal_completion_unchanged(self):
        """Sanity: a normal run that submits on a non-final turn still works
        and never sees the FINAL TURN nudge."""
        responses = [
            _wrap_code("x = 21 * 2\nprint(x)"),
            _wrap_code("SUBMIT(answer=42)"),
        ]
        lm = ScriptedLM(responses)
        rlm = RLM.from_task("Compute 21*2", outputs=["answer"], lm=lm,
                            max_turns=4, timeout=5)
        result = rlm.run()
        assert result.submitted
        assert result.payload == {"answer": 42}
        # Submitted on turn 2 of 4 → no LM call ever saw the nudge
        for msgs in lm.messages:
            assert "FINAL TURN" not in _last_user_msg(msgs)
