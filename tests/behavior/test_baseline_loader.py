"""Tests for the baseline loader/validator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .baseline_loader import (
    BaselineSchemaError,
    SUITE_VERSION,
    evaluate_gates,
    load_baseline,
    parse_baseline,
)


def _valid_payload() -> dict:
    return {
        "suite_version": SUITE_VERSION,
        "calibrated_at": "2026-05-07T12:00:00Z",
        "calibrated_against_commit": "deadbeef",
        "questions_sha256": "0" * 64,
        "max_turns": 8,
        "timeout_s": 120,
        "calibration_runs_per_qid": 5,
        "models": {
            "openai/gpt-4.1-mini": {
                "questions": {
                    "C1": {"baseline_pass_rate": 1.0, "passes": 5, "runs": 5, "expected_to_pass": True},
                    "C5": {"baseline_pass_rate": 1.0, "passes": 5, "runs": 5, "expected_to_pass": True},
                    "M1": {"baseline_pass_rate": 0.8, "passes": 4, "runs": 5, "expected_to_pass": True},
                    "M3": {"baseline_pass_rate": 0.6, "passes": 3, "runs": 5, "expected_to_pass": False},
                    "S5": {"baseline_pass_rate": 1.0, "passes": 5, "runs": 5, "expected_to_pass": True},
                },
                "aggregate": {"baseline_passes": 4, "min_passes": 3},
            }
        },
    }


# ---------------------------------------------------------------------------
# parse_baseline
# ---------------------------------------------------------------------------

class TestParseBaseline:
    def test_valid_payload_parses(self) -> None:
        b = parse_baseline(_valid_payload())
        assert b.suite_version == SUITE_VERSION
        assert "openai/gpt-4.1-mini" in b.models
        mb = b.models["openai/gpt-4.1-mini"]
        assert mb.baseline_passes == 4
        assert mb.min_passes == 3
        assert sorted(mb.stable_qids()) == ["C1", "C5", "M1", "S5"]

    def test_missing_top_key_raises(self) -> None:
        p = _valid_payload()
        del p["questions_sha256"]
        with pytest.raises(BaselineSchemaError, match="questions_sha256"):
            parse_baseline(p)

    def test_wrong_suite_version_raises(self) -> None:
        p = _valid_payload()
        p["suite_version"] = "behavior-v0"
        with pytest.raises(BaselineSchemaError, match="suite_version mismatch"):
            parse_baseline(p)

    def test_empty_models_raises(self) -> None:
        p = _valid_payload()
        p["models"] = {}
        with pytest.raises(BaselineSchemaError, match="non-empty"):
            parse_baseline(p)

    def test_empty_questions_raises(self) -> None:
        p = _valid_payload()
        p["models"]["openai/gpt-4.1-mini"]["questions"] = {}
        with pytest.raises(BaselineSchemaError, match="non-empty"):
            parse_baseline(p)

    def test_min_passes_above_baseline_raises(self) -> None:
        p = _valid_payload()
        p["models"]["openai/gpt-4.1-mini"]["aggregate"]["min_passes"] = 99
        with pytest.raises(BaselineSchemaError, match="min_passes"):
            parse_baseline(p)

    def test_aggregate_inconsistent_with_per_qid_raises(self) -> None:
        # baseline_passes = 4 but only 3 expected_to_pass qids.
        p = _valid_payload()
        p["models"]["openai/gpt-4.1-mini"]["questions"]["S5"]["expected_to_pass"] = False
        with pytest.raises(BaselineSchemaError, match="expected_to_pass"):
            parse_baseline(p)


# ---------------------------------------------------------------------------
# load_baseline
# ---------------------------------------------------------------------------

class TestLoadBaseline:
    def test_missing_file_raises_with_calibration_hint(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="--calibrate"):
            load_baseline(tmp_path / "nonexistent.json")

    def test_invalid_json_raises_schema_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(BaselineSchemaError, match="invalid JSON"):
            load_baseline(bad)

    def test_round_trip(self, tmp_path: Path) -> None:
        f = tmp_path / "baselines.json"
        f.write_text(json.dumps(_valid_payload()), encoding="utf-8")
        b = load_baseline(f)
        assert "openai/gpt-4.1-mini" in b.models


# ---------------------------------------------------------------------------
# evaluate_gates
# ---------------------------------------------------------------------------

class TestEvaluateGates:
    def _mb(self):
        return parse_baseline(_valid_payload()).models["openai/gpt-4.1-mini"]

    def test_all_stable_passing_passes(self) -> None:
        mb = self._mb()
        outcome = evaluate_gates(mb, {"C1": True, "C5": True, "M1": True, "M3": False, "S5": True})
        assert outcome.passed
        assert outcome.reasons == []

    def test_per_qid_regression_fails_even_if_aggregate_ok(self) -> None:
        # baseline_passes=4 min_passes=3. Drop C1 (stable), gain M3 (not expected). Aggregate=4 still.
        mb = self._mb()
        outcome = evaluate_gates(mb, {"C1": False, "C5": True, "M1": True, "M3": True, "S5": True})
        assert not outcome.passed
        assert any("C1" in r and "regressed" in r for r in outcome.reasons)

    def test_missing_qid_treated_as_per_qid_failure(self) -> None:
        mb = self._mb()
        outcome = evaluate_gates(mb, {"C5": True, "M1": True, "S5": True})
        assert not outcome.passed
        assert any("C1" in r and "missing" in r for r in outcome.reasons)

    def test_aggregate_floor_violation_fails(self) -> None:
        mb = self._mb()
        # Drop two stable qids; per-qid will trip too, but aggregate would also.
        outcome = evaluate_gates(mb, {"C1": False, "C5": False, "M1": True, "M3": False, "S5": True})
        assert not outcome.passed
        assert any("aggregate" in r for r in outcome.reasons)

    def test_one_stable_drop_with_otherwise_full_pass_fails_per_qid(self) -> None:
        # The motivating regression class: 5/5 baseline, PR shows 4/5 with one stable qid lost.
        mb = self._mb()
        outcome = evaluate_gates(mb, {"C1": True, "C5": True, "M1": False, "M3": False, "S5": True})
        # Aggregate: 3 passes, floor is 3 -> aggregate OK. Per-qid: M1 was stable -> fail.
        assert not outcome.passed
        assert any("M1" in r and "regressed" in r for r in outcome.reasons)
