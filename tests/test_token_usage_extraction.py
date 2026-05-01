"""Tests for token-usage extraction from LM calls.

The custom v6 engine calls each LM via ``_call_lm_with_meta`` and feeds the
returned ``raw_response`` to ``_extract_usage``. ``dspy.LM.__call__``
returns only the completion text (a list of strings), so the extractor
must also peek at ``lm.history[-1]`` — without that fallback every turn's
``prompt_tokens`` ends up ``None`` and the engine cannot report cost.

These tests pin the expected behavior so the tier notebooks (and the
bandit's cost-savings claims that depend on them) actually report
non-None token counts.
"""

from __future__ import annotations

from fabric_rlm.runtime import _call_lm_with_meta, _extract_usage, _usage_field


class _FakeDspyLM:
    """Mimics ``dspy.LM``'s public surface: callable + ``history`` list."""

    def __init__(self, response_text="ok", usage=None):
        self.response_text = response_text
        self.usage = usage
        self.history: list[dict] = []

    def __call__(self, messages=None, **kwargs):
        # dspy.LM appends to history then returns ``[completion_text]``
        entry = {
            "prompt": None,
            "messages": messages,
            "kwargs": kwargs,
            "response": object(),  # opaque
            "outputs": [self.response_text],
            "usage": dict(self.usage) if self.usage else {},
        }
        self.history.append(entry)
        return [self.response_text]


def test_extract_usage_finds_usage_in_history_entry():
    """When raw_response is a dict with a 'usage' key, extract it."""
    entry = {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
    usage = _extract_usage(entry)
    assert usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


def test_call_lm_with_meta_falls_back_to_history():
    """The dspy-style fake returns a list[str] from __call__ but stores
    usage on history[-1]. ``_call_lm_with_meta`` must surface the history
    entry as the raw_response so ``_extract_usage`` can find the usage."""
    lm = _FakeDspyLM(response_text="ok", usage={"prompt_tokens": 11, "completion_tokens": 22})
    text, raw, elapsed = _call_lm_with_meta(lm, [{"role": "user", "content": "hi"}])
    assert text == "ok"
    assert isinstance(raw, dict)
    assert raw is lm.history[-1]
    usage = _extract_usage(raw)
    assert _usage_field(usage, "prompt_tokens") == 11
    assert _usage_field(usage, "completion_tokens") == 22
    assert elapsed >= 0


def test_call_lm_with_meta_no_history_returns_response():
    """Backends without a ``history`` attribute (e.g. raw callables) should
    fall through to returning the literal response object."""
    def callable_lm(messages=None, **kw):
        return "raw text"

    text, raw, elapsed = _call_lm_with_meta(callable_lm, [{"role": "user", "content": "x"}])
    assert text == "raw text"
    assert raw == "raw text"


def test_call_lm_with_meta_no_history_growth_returns_response():
    """If a dspy-style backend has a ``history`` attribute but doesn't append
    an entry (e.g. ``settings.disable_history``), do not falsely re-use a
    pre-existing entry from a prior call."""
    lm = _FakeDspyLM(response_text="cached")
    lm.history.append({"usage": {"prompt_tokens": 99}})  # stale entry from before
    # Override __call__ to not append — simulating disable_history
    def stub_call(messages=None, **kw):
        return ["cached"]
    lm.__call__ = stub_call  # type: ignore[assignment]
    text, raw, elapsed = _call_lm_with_meta(lm, [{"role": "user", "content": "x"}])
    assert text == "cached"
    # Must NOT return the stale entry (history didn't grow)
    assert raw != lm.history[0]


def test_aggregation_returns_none_when_no_turn_has_usage():
    """Sanity: the aggregator reports None (not 0) when no turn has usage."""
    from fabric_rlm.runtime import _aggregate_trajectory_metrics
    from fabric_rlm.trajectory import Trajectory, TurnRecord

    traj = Trajectory()
    traj.turns.append(TurnRecord(turn=0, code="", stdout="", stderr="", error=None,
                                  submitted=False, state={},
                                  prompt_tokens=None, completion_tokens=None))
    out = _aggregate_trajectory_metrics(traj)
    assert out["total_prompt_tokens"] is None
    assert out["total_completion_tokens"] is None


def test_aggregation_sums_present_usage():
    """Aggregator sums turn usage even when some turns are None."""
    from fabric_rlm.runtime import _aggregate_trajectory_metrics
    from fabric_rlm.trajectory import Trajectory, TurnRecord

    def _t(pt, ct):
        return TurnRecord(turn=0, code="", stdout="", stderr="", error=None,
                          submitted=False, state={},
                          prompt_tokens=pt, completion_tokens=ct)
    traj = Trajectory()
    traj.turns.append(_t(10, 20))
    traj.turns.append(_t(None, None))
    traj.turns.append(_t(5, 15))
    out = _aggregate_trajectory_metrics(traj)
    assert out["total_prompt_tokens"] == 15
    assert out["total_completion_tokens"] == 35
