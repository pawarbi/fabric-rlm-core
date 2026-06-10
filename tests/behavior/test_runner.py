"""Tests for the runner module (offline; no real LM calls).

Network-dependent behavior is exercised by ``test_behavior_baseline.py`` when
``OPENROUTER_API_KEY`` is set.  These tests cover the pure-Python pieces:
classification, calibration aggregation, baseline merging, CLI argument
parsing.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from . import runner as runner_mod
from .questions import QUESTIONS, Question
from .runner import (
    IncompatibleBaselineMerge,
    QuestionRun,
    _classify_error,
    _is_reasoning,
    calibrate,
    main,
    merge_calibration,
    run_question,
)


# ---------------------------------------------------------------------------
# error classification
# ---------------------------------------------------------------------------

class TestClassifyError:
    @pytest.mark.parametrize(
        "msg",
        [
            "HTTPError: 429 Too Many Requests",
            "Timeout: read timed out",
            "ConnectionError: connection reset by peer",
            "APIError: 503 Service Unavailable",
            "Exception: rate limit exceeded for model",
            "Exception: 502 Bad Gateway",
        ],
    )
    def test_infra_messages_classified_as_infra(self, msg: str) -> None:
        e = RuntimeError(msg)
        assert _classify_error(e) == "infra"

    @pytest.mark.parametrize(
        "msg",
        [
            "ValueError: bad answer format",
            "KeyError: 'numbers'",
            "AttributeError: 'NoneType' has no attribute 'outputs'",
        ],
    )
    def test_non_infra_messages_classified_as_runner_error(self, msg: str) -> None:
        e = RuntimeError(msg)
        assert _classify_error(e) == "runner_error"


# ---------------------------------------------------------------------------
# reasoning model detection
# ---------------------------------------------------------------------------

class TestIsReasoning:
    @pytest.mark.parametrize(
        "model",
        ["openai/gpt-5", "openai/gpt-5.4-nano", "openai/o1-mini", "openai/o3", "openai/o4-pro"],
    )
    def test_reasoning_models_detected(self, model: str) -> None:
        assert _is_reasoning(model)

    @pytest.mark.parametrize(
        "model",
        ["openai/gpt-4.1-mini", "openai/gpt-4o", "deepseek/deepseek-chat", "anthropic/claude-3.5"],
    )
    def test_non_reasoning_models_not_detected(self, model: str) -> None:
        assert not _is_reasoning(model)


# ---------------------------------------------------------------------------
# calibrate (using a stub run_question)
# ---------------------------------------------------------------------------

def _make_stub_runner(scripted: dict[str, list[bool]]):
    """Return a stub for run_question that returns scripted pass/fail per qid.

    ``scripted`` maps qid -> list[bool] of pass outcomes; consumed in order.
    """
    counters: dict[str, int] = {qid: 0 for qid in scripted}

    def stub(q: Question, model: str, **kw) -> QuestionRun:
        i = counters[q.qid]
        counters[q.qid] += 1
        passed = scripted[q.qid][i]
        return QuestionRun(
            qid=q.qid,
            model=model,
            passed=passed,
            answer=q.expected if passed else "wrong",
            expected=q.expected,
            reason="" if passed else "stubbed wrong answer",
            error_class=None if passed else "wrong_answer",
            n_turns=1,
            elapsed_s=0.001,
            attempts=1,
        )

    return stub


class TestCalibrate:
    def test_all_pass_makes_all_expected_to_pass(self) -> None:
        scripted = {q.qid: [True, True, True] for q in QUESTIONS}
        with patch.object(runner_mod, "run_question", _make_stub_runner(scripted)):
            payload = calibrate("openai/gpt-4.1-mini", runs=3)
        mb = payload["models"]["openai/gpt-4.1-mini"]
        assert all(qb["expected_to_pass"] for qb in mb["questions"].values())
        assert mb["aggregate"]["baseline_passes"] == 5
        assert mb["aggregate"]["min_passes"] == 4

    def test_partial_pass_below_threshold_is_not_expected(self) -> None:
        # 4/5 pass rate is below the default 1.0 threshold -> expected_to_pass=False.
        scripted = {q.qid: [True, True, True, True, False] for q in QUESTIONS}
        with patch.object(runner_mod, "run_question", _make_stub_runner(scripted)):
            payload = calibrate("openai/gpt-4.1-mini", runs=5)
        mb = payload["models"]["openai/gpt-4.1-mini"]
        for qb in mb["questions"].values():
            assert qb["expected_to_pass"] is False
            assert qb["passes"] == 4
            assert qb["runs"] == 5
        assert mb["aggregate"]["baseline_passes"] == 0
        assert mb["aggregate"]["min_passes"] == 0

    def test_pass_rate_at_threshold_is_expected(self) -> None:
        # 5/5 = 1.0 exactly -> expected_to_pass=True.
        scripted = {q.qid: [True, True, True, True, True] for q in QUESTIONS}
        with patch.object(runner_mod, "run_question", _make_stub_runner(scripted)):
            payload = calibrate("openai/gpt-4.1-mini", runs=5)
        mb = payload["models"]["openai/gpt-4.1-mini"]
        assert all(qb["expected_to_pass"] for qb in mb["questions"].values())

    def test_explicit_lower_threshold_promotes_4_of_5(self) -> None:
        # Operators can opt back into the looser threshold.
        scripted = {q.qid: [True, True, True, True, False] for q in QUESTIONS}
        with patch.object(runner_mod, "run_question", _make_stub_runner(scripted)):
            payload = calibrate("openai/gpt-4.1-mini", runs=5, pass_rate_threshold=0.8)
        mb = payload["models"]["openai/gpt-4.1-mini"]
        assert all(qb["expected_to_pass"] for qb in mb["questions"].values())

    def test_payload_includes_questions_sha256_and_metadata(self) -> None:
        scripted = {q.qid: [True] for q in QUESTIONS}
        with patch.object(runner_mod, "run_question", _make_stub_runner(scripted)):
            payload = calibrate("openai/gpt-4.1-mini", runs=1)
        assert "questions_sha256" in payload and len(payload["questions_sha256"]) == 64
        assert payload["max_turns"] == 8
        assert payload["timeout_s"] == 120
        assert payload["calibration_runs_per_qid"] == 1


# ---------------------------------------------------------------------------
# merge_calibration
# ---------------------------------------------------------------------------

class TestMergeCalibration:
    def test_merge_into_empty_returns_new(self) -> None:
        new = {"suite_version": "behavior-v1", "models": {"a": {"x": 1}}, "calibrated_at": "t",
               "calibrated_against_commit": "c", "questions_sha256": "h", "max_turns": 8,
               "timeout_s": 120, "calibration_runs_per_qid": 5}
        assert merge_calibration(None, new) == new
        assert merge_calibration({}, new) == new

    def test_merge_preserves_other_models(self) -> None:
        existing = {
            "suite_version": "behavior-v1", "calibrated_at": "old", "calibrated_against_commit": "old",
            "questions_sha256": "h", "max_turns": 8, "timeout_s": 120, "calibration_runs_per_qid": 5,
            "models": {"openai/gpt-4.1-mini": {"questions": {"C1": {}}, "aggregate": {}}},
        }
        new = {
            "suite_version": "behavior-v1", "calibrated_at": "new", "calibrated_against_commit": "new",
            "questions_sha256": "h", "max_turns": 8, "timeout_s": 120, "calibration_runs_per_qid": 5,
            "models": {"deepseek/deepseek-chat:free": {"questions": {"C1": {}}, "aggregate": {}}},
        }
        merged = merge_calibration(existing, new)
        assert "openai/gpt-4.1-mini" in merged["models"]
        assert "deepseek/deepseek-chat:free" in merged["models"]
        # calibrated_at still reflects the most recent calibration run.
        assert merged["calibrated_at"] == "new"
        assert merged["questions_sha256"] == "h"

    def test_merge_replaces_same_model(self) -> None:
        existing = {
            "suite_version": "behavior-v1", "calibrated_at": "old", "calibrated_against_commit": "old",
            "questions_sha256": "h", "max_turns": 8, "timeout_s": 120, "calibration_runs_per_qid": 5,
            "models": {"openai/gpt-4.1-mini": {"questions": {"OLD": {}}, "aggregate": {"baseline_passes": 0, "min_passes": 0}}},
        }
        new = {
            "suite_version": "behavior-v1", "calibrated_at": "new", "calibrated_against_commit": "new",
            "questions_sha256": "h", "max_turns": 8, "timeout_s": 120, "calibration_runs_per_qid": 5,
            "models": {"openai/gpt-4.1-mini": {"questions": {"NEW": {}}, "aggregate": {"baseline_passes": 1, "min_passes": 0}}},
        }
        merged = merge_calibration(existing, new)
        assert "NEW" in merged["models"]["openai/gpt-4.1-mini"]["questions"]
        assert "OLD" not in merged["models"]["openai/gpt-4.1-mini"]["questions"]


class TestMergeGuard:
    """Merging into a baseline whose suite metadata disagrees should refuse by default."""

    def _existing(self, **overrides: Any) -> dict[str, Any]:
        base = {
            "suite_version": "behavior-v1",
            "calibrated_at": "old", "calibrated_against_commit": "old",
            "questions_sha256": "OLD_HASH",
            "max_turns": 8, "timeout_s": 120, "calibration_runs_per_qid": 5,
            "models": {"openai/gpt-4.1-mini": {"questions": {"C1": {}}, "aggregate": {"baseline_passes": 0, "min_passes": 0}}},
        }
        base.update(overrides)
        return base

    def _new(self, **overrides: Any) -> dict[str, Any]:
        base = {
            "suite_version": "behavior-v1",
            "calibrated_at": "new", "calibrated_against_commit": "new",
            "questions_sha256": "OLD_HASH",
            "max_turns": 8, "timeout_s": 120, "calibration_runs_per_qid": 5,
            "models": {"deepseek/deepseek-chat-v3.1:free": {"questions": {}, "aggregate": {"baseline_passes": 0, "min_passes": 0}}},
        }
        base.update(overrides)
        return base

    @pytest.mark.parametrize("key,bad_value", [
        ("questions_sha256", "NEW_HASH"),
        ("max_turns", 16),
        ("timeout_s", 240),
        ("calibration_runs_per_qid", 3),
        ("suite_version", "behavior-v2"),
    ])
    def test_merge_refused_on_metadata_mismatch(self, key: str, bad_value: Any) -> None:
        with pytest.raises(IncompatibleBaselineMerge, match=key):
            merge_calibration(self._existing(), self._new(**{key: bad_value}))

    def test_force_merge_overrides_guard_and_uses_new_metadata(self) -> None:
        merged = merge_calibration(
            self._existing(),
            self._new(questions_sha256="NEW_HASH", max_turns=16),
            force=True,
        )
        assert merged["questions_sha256"] == "NEW_HASH"
        assert merged["max_turns"] == 16
        # Other model is preserved.
        assert "openai/gpt-4.1-mini" in merged["models"]
        assert "deepseek/deepseek-chat-v3.1:free" in merged["models"]

    def test_compatible_merge_succeeds(self) -> None:
        merged = merge_calibration(self._existing(), self._new())
        assert "openai/gpt-4.1-mini" in merged["models"]
        assert "deepseek/deepseek-chat-v3.1:free" in merged["models"]
        # Sha unchanged because both agreed.
        assert merged["questions_sha256"] == "OLD_HASH"


# ---------------------------------------------------------------------------
# run_question retry contract
# ---------------------------------------------------------------------------

class TestRunQuestionRetry:
    """Direct verification of the retry-once-on-infra contract."""

    def _q(self) -> Question:
        return QUESTIONS[0]

    def _patch_run_once(self, sequence: list[tuple[Any, BaseException | None]]):
        """Return a stub _run_once that yields scripted (answer, exc) tuples."""
        calls = {"n": 0}

        def stub(q: Question, model: str, *, max_turns: int, timeout_s: float):
            i = calls["n"]
            calls["n"] += 1
            ans, exc = sequence[i]
            return (ans, 1, 0.001, exc)

        return stub, calls

    def test_infra_then_pass_two_attempts_recovered(self) -> None:
        q = self._q()
        # First call raises infra; second returns the correct answer.
        stub, calls = self._patch_run_once([
            (None, RuntimeError("HTTP 429 rate limit")),
            (q.expected, None),
        ])
        with patch.object(runner_mod, "_run_once", stub), patch.object(runner_mod.time, "sleep", lambda _s: None):
            res = run_question(q, "openai/gpt-4.1-mini")
        assert calls["n"] == 2
        assert res.passed is True
        assert res.attempts == 2

    def test_infra_then_infra_two_attempts_final_infra_failure(self) -> None:
        q = self._q()
        stub, calls = self._patch_run_once([
            (None, RuntimeError("HTTP 503 service unavailable")),
            (None, RuntimeError("HTTP 502 bad gateway")),
        ])
        with patch.object(runner_mod, "_run_once", stub), patch.object(runner_mod.time, "sleep", lambda _s: None):
            res = run_question(q, "openai/gpt-4.1-mini")
        assert calls["n"] == 2
        assert res.passed is False
        assert res.error_class == "infra"
        assert res.attempts == 2

    def test_runner_error_no_retry(self) -> None:
        q = self._q()
        stub, calls = self._patch_run_once([
            (None, RuntimeError("ValueError: bad answer format")),
        ])
        with patch.object(runner_mod, "_run_once", stub):
            res = run_question(q, "openai/gpt-4.1-mini")
        assert calls["n"] == 1
        assert res.passed is False
        assert res.error_class == "runner_error"
        assert res.attempts == 1

    def test_wrong_answer_no_retry(self) -> None:
        q = self._q()
        # Successful run returning a wrong answer.
        stub, calls = self._patch_run_once([
            ("wrong_answer", None),
        ])
        with patch.object(runner_mod, "_run_once", stub):
            res = run_question(q, "openai/gpt-4.1-mini")
        assert calls["n"] == 1
        assert res.passed is False
        assert res.error_class == "wrong_answer"
        assert res.attempts == 1

    def test_retry_disabled_takes_one_shot(self) -> None:
        q = self._q()
        stub, calls = self._patch_run_once([
            (None, RuntimeError("HTTP 429 rate limit")),
        ])
        with patch.object(runner_mod, "_run_once", stub):
            res = run_question(q, "openai/gpt-4.1-mini", retry_on_infra=False)
        assert calls["n"] == 1
        assert res.error_class == "infra"
        assert res.attempts == 1


# ---------------------------------------------------------------------------
# status-code classification
# ---------------------------------------------------------------------------

class TestClassifyByStatusCode:
    @pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504, 520, 522, 529])
    def test_status_code_attribute_classified_as_infra(self, code: int) -> None:
        e = RuntimeError("opaque message")
        e.status_code = code  # type: ignore[attr-defined]
        assert _classify_error(e) == "infra"

    def test_status_code_via_response_attribute(self) -> None:
        class _Resp:
            status_code = 503
        e = RuntimeError("opaque")
        e.response = _Resp()  # type: ignore[attr-defined]
        assert _classify_error(e) == "infra"

    def test_4xx_other_than_listed_is_runner_error(self) -> None:
        e = RuntimeError("opaque")
        e.status_code = 404  # type: ignore[attr-defined]
        assert _classify_error(e) == "runner_error"

    def test_401_and_403_classify_as_auth(self) -> None:
        for status in (401, 403):
            e = RuntimeError("opaque")
            e.status_code = status  # type: ignore[attr-defined]
            assert _classify_error(e) == "auth"

    def test_auth_message_tokens_classify_as_auth(self) -> None:
        e = RuntimeError(
            "litellm.AuthenticationError: OpenrouterException - "
            '{"error":{"message":"User not found.","code":401}}'
        )
        assert _classify_error(e) == "auth"


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

class TestCli:
    def test_no_args_prints_help_and_returns_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main([])
        assert rc == 2
        out = capsys.readouterr().out
        assert "behavior-runner" in out

    def test_calibrate_writes_file(self, tmp_path: Path) -> None:
        scripted = {q.qid: [True] for q in QUESTIONS}
        out = tmp_path / "baselines.json"
        with patch.object(runner_mod, "run_question", _make_stub_runner(scripted)):
            rc = main([
                "--calibrate", "--model", "openai/gpt-4.1-mini",
                "--runs", "1", "--out", str(out),
            ])
        assert rc == 0
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["suite_version"] == "behavior-v1"
        assert "openai/gpt-4.1-mini" in payload["models"]


# ---------------------------------------------------------------------------
# Engine override env var (Phase 3 safety net for engine-consolidation)
# ---------------------------------------------------------------------------

class TestEngineOverride:
    """``BEHAVIOR_CI_ENGINE_OVERRIDE`` lets us run the suite with a non-default
    engine without changing the runner's stable contract. Used by Phase 3 of
    the engine-consolidation plan to verify ``engine="auto"`` is byte-equivalent
    to the current default for the behavior-CI workload.

    Contract:
    * Unset (or empty) → no ``engine=`` kwarg passed; default is preserved.
    * Set to a non-empty string → that string is forwarded to ``RLM.from_task``.
    """

    @staticmethod
    def _stub_rlm_module():
        """Build a fake ``fabric_rlm`` module exposing an ``RLM`` whose
        ``from_task`` records its kwargs. Returns (fake_module, calls_list)."""
        import sys
        import types

        calls: list[dict[str, Any]] = []

        class _FakeResult:
            outputs = {"answer": "ok"}
            n_turns = 1

        class _FakeRLM:
            @classmethod
            def from_task(cls, **kwargs: Any) -> "_FakeRLM":
                calls.append(dict(kwargs))
                return cls()

            def run(self) -> _FakeResult:
                return _FakeResult()

        fake = types.ModuleType("fabric_rlm")
        fake.RLM = _FakeRLM  # type: ignore[attr-defined]
        return fake, calls, sys

    def _run_with_env(self, env_value: str | None) -> dict[str, Any]:
        fake_mod, calls, sys_mod = self._stub_rlm_module()
        q = QUESTIONS[0]
        with patch.dict(sys_mod.modules, {"fabric_rlm": fake_mod}):
            with patch.object(runner_mod, "make_lm", lambda _model: object()):
                env_patch: dict[str, str] = {}
                if env_value is not None:
                    env_patch["BEHAVIOR_CI_ENGINE_OVERRIDE"] = env_value
                with patch.dict("os.environ", env_patch, clear=False):
                    if env_value is None:
                        # Ensure unset even if shell exported it.
                        import os as _os
                        _os.environ.pop("BEHAVIOR_CI_ENGINE_OVERRIDE", None)
                    runner_mod._run_once(q, "openai/gpt-4.1-mini", max_turns=1, timeout_s=10.0)
        assert len(calls) == 1, f"expected 1 RLM construction, got {len(calls)}"
        return calls[0]

    def test_env_unset_omits_engine_kwarg(self) -> None:
        kwargs = self._run_with_env(None)
        assert "engine" not in kwargs

    def test_env_empty_string_omits_engine_kwarg(self) -> None:
        kwargs = self._run_with_env("")
        assert "engine" not in kwargs

    @pytest.mark.parametrize("value", ["auto", "v6-custom", "v7-dspy", "default", "dspy"])
    def test_env_value_forwarded_as_engine_kwarg(self, value: str) -> None:
        kwargs = self._run_with_env(value)
        assert kwargs.get("engine") == value

    def test_env_value_is_stripped_before_forwarding(self) -> None:
        """Whitespace-only is treated as unset; padded values are stripped."""
        kwargs = self._run_with_env("   ")
        assert "engine" not in kwargs

        kwargs = self._run_with_env("  auto  ")
        assert kwargs.get("engine") == "auto"
