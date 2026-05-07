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
    QuestionRun,
    _classify_error,
    _is_reasoning,
    calibrate,
    main,
    merge_calibration,
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
        # 2/5 pass rate is below the 0.8 threshold -> expected_to_pass=False.
        scripted = {q.qid: [True, True, False, False, False] for q in QUESTIONS}
        with patch.object(runner_mod, "run_question", _make_stub_runner(scripted)):
            payload = calibrate("openai/gpt-4.1-mini", runs=5)
        mb = payload["models"]["openai/gpt-4.1-mini"]
        for qb in mb["questions"].values():
            assert qb["expected_to_pass"] is False
            assert qb["passes"] == 2
            assert qb["runs"] == 5
        assert mb["aggregate"]["baseline_passes"] == 0
        assert mb["aggregate"]["min_passes"] == 0

    def test_pass_rate_at_threshold_is_expected(self) -> None:
        # 4/5 = 0.8 exactly -> expected_to_pass=True.
        scripted = {q.qid: [True, True, True, True, False] for q in QUESTIONS}
        with patch.object(runner_mod, "run_question", _make_stub_runner(scripted)):
            payload = calibrate("openai/gpt-4.1-mini", runs=5)
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
            "questions_sha256": "h2", "max_turns": 8, "timeout_s": 120, "calibration_runs_per_qid": 5,
            "models": {"deepseek/deepseek-chat:free": {"questions": {"C1": {}}, "aggregate": {}}},
        }
        merged = merge_calibration(existing, new)
        assert "openai/gpt-4.1-mini" in merged["models"]
        assert "deepseek/deepseek-chat:free" in merged["models"]
        # Top-level metadata reflects the new run.
        assert merged["calibrated_at"] == "new"
        assert merged["questions_sha256"] == "h2"

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
