"""Code-extraction robustness for the dspy turn loop.

Two related fixes to how a model's response is turned into runnable code:

* **No runnable block -> don't execute prose.** When the response has no fenced
  block and the bare text isn't valid Python, the runtime must NOT ship the
  prose to the worker (where it can only come back as a SyntaxError, burning a
  worker round-trip and a turn). It mirrors the existing truncated-fence guard:
  short-circuit with a clear "resend one complete ```python block" message.

* **Prefer the model's final code block.** Models that think out loud put a
  sketch in the first block and the corrected code in a later block; some also
  append an *expected output* block. Extraction should pick the model's last
  language-tagged, parseable block rather than the first fence by position.
"""

from __future__ import annotations

from fabric_rlm import RLM
from fabric_rlm.interpreter import ExecResult
from fabric_rlm.runtime import _select_code_block


class ScriptedLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def __call__(self, *, messages):
        self.messages.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("No scripted responses left")
        return self.responses.pop(0)


def _wrap(body: str, lang: str = "python") -> str:
    return f"```{lang}\n{body}\n```"


def _last_user_msg(messages):
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")


# ---------------------------------------------------------------------------
# Unit: _select_code_block
# ---------------------------------------------------------------------------


class TestSelectCodeBlock:
    def test_single_python_block(self):
        assert _select_code_block(_wrap("print(1)")) == "print(1)"

    def test_prefers_last_python_block_on_revision(self):
        """Sketch in block 1, corrected code in block 2 -> take block 2."""
        text = (
            "Here's a rough idea:\n"
            + _wrap("x = bad_sketch(")
            + "\nWait, let me fix that:\n"
            + _wrap("x = 1 + 1\nSUBMIT(answer=x)")
        )
        assert _select_code_block(text) == "x = 1 + 1\nSUBMIT(answer=x)"

    def test_code_then_expected_output_picks_the_code(self):
        """Real code in a ```python block, then an *untagged* output example.
        Must run the code, not the example."""
        text = (
            _wrap("SUBMIT(answer=42)")
            + "\nExpected output:\n"
            + _wrap("42", lang="")
        )
        assert _select_code_block(text) == "SUBMIT(answer=42)"

    def test_falls_back_to_last_block_when_none_parse(self):
        """All tagged blocks are broken -> still return the last one so the
        worker surfaces a real SyntaxError on the model's latest attempt."""
        text = _wrap("def f(") + "\n" + _wrap("def g(")
        assert _select_code_block(text) == "def g("

    def test_bare_valid_code_without_fence_is_kept(self):
        """Lenient fallback: a fence-less response that IS valid Python still
        runs (don't punish models that skip the fence but send real code)."""
        assert _select_code_block("SUBMIT(answer='ok')") == "SUBMIT(answer='ok')"

    def test_pure_prose_returns_none(self):
        assert _select_code_block("Sure, let me think about how to do this.") is None

    def test_empty_returns_none(self):
        assert _select_code_block("   \n  ") is None

    def test_untagged_fence_with_code_is_used_when_no_tagged_block(self):
        assert _select_code_block("```\nSUBMIT(answer=1)\n```") == "SUBMIT(answer=1)"


# ---------------------------------------------------------------------------
# Integration: the turn loop must not execute prose
# ---------------------------------------------------------------------------


class TestNoCodeBlockShortCircuit:
    def test_fenceless_prose_skips_execution_and_asks_for_a_block(self):
        responses = [
            "I'll start by reasoning about the inputs, then compute the result.",
            _wrap("SUBMIT(answer='ok')"),
        ]
        lm = ScriptedLM(responses)
        rlm = RLM.from_task("Whatever", outputs=["answer"], lm=lm,
                            max_turns=3, timeout=5)
        result = rlm.run()

        # The prose turn must have produced a "resend a code block" nudge, NOT a
        # SyntaxError from executing the prose.
        assert len(lm.messages) >= 2
        nudge = _last_user_msg(lm.messages[1])
        assert "code block" in nudge.lower()
        assert "SyntaxError" not in nudge
        assert "truncated" not in nudge.lower()
        # And the run still succeeds on the following (real) turn.
        assert result.submitted is True
        assert result.outputs.get("answer") == "ok"

    def test_no_code_nudge_on_final_turn_includes_final_suffix(self):
        responses = [
            "Just some prose, no code at all here.",
            _wrap("SUBMIT(answer='ok')"),
        ]
        lm = ScriptedLM(responses)
        rlm = RLM.from_task("Whatever", outputs=["answer"], lm=lm,
                            max_turns=2, timeout=5)
        rlm.run()
        nudge = _last_user_msg(lm.messages[1])
        assert "code block" in nudge.lower()
        assert "FINAL TURN" in nudge

    def test_bare_code_response_executes_without_a_nudge(self):
        """A fence-less but valid-Python response runs immediately (no nudge)."""
        lm = ScriptedLM([_wrap("SUBMIT(answer='done')").replace("```python\n", "").replace("\n```", "")])
        rlm = RLM.from_task("Whatever", outputs=["answer"], lm=lm,
                            max_turns=2, timeout=5)
        result = rlm.run()
        assert result.submitted is True
        assert len(lm.messages) == 1  # submitted on the first turn, no retry
