"""NEW-J: cleanup of dead `cost` field on TurnRecord.

`cost` was declared on `TurnRecord` with a `field(default_factory=dict)` but
never written by the runtime, never read by the library, never asserted by any
test. Across 1354 turns of the SSB benchmark it was emitted as `"cost": {}`
on every record, bloating JSONL output and confusing downstream consumers
that assumed the empty dict was a populated-but-zero cost report.

This test file pins the contract that `cost` is no longer part of the public
`TurnRecord` API. If we ever re-introduce per-turn cost tracking, it should
land with explicit population logic and tests, not as a silent default-empty
dict.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from fabric_rlm.trajectory import Trajectory, TurnRecord


def _make_turn(**overrides) -> TurnRecord:
    base = dict(
        turn=1,
        code="x = 1",
        stdout="",
        stderr="",
        error=None,
        submitted=False,
        state={},
    )
    base.update(overrides)
    return TurnRecord(**base)


class TestCostFieldRemoved:
    def test_cost_not_in_dataclass_fields(self):
        names = {f.name for f in dataclasses.fields(TurnRecord)}
        assert "cost" not in names

    def test_to_dict_does_not_include_cost(self):
        record = _make_turn()
        assert "cost" not in record.to_dict()

    def test_jsonl_does_not_emit_cost_key(self):
        traj = Trajectory()
        traj.append(_make_turn())
        traj.append(_make_turn(turn=2, code="y = 2"))
        for line in traj.to_jsonl().splitlines():
            obj = json.loads(line)
            assert "cost" not in obj

    def test_constructing_with_cost_kwarg_raises(self):
        # If a caller still tries to pass cost=..., they get a clear TypeError
        # instead of silently storing an unused value.
        with pytest.raises(TypeError):
            _make_turn(cost={"usd": 0.0})


class TestSurvivingFieldsIntact:
    """Sanity: the cleanup doesn't accidentally also remove live fields."""

    @pytest.mark.parametrize(
        "field_name",
        [
            "turn",
            "code",
            "stdout",
            "stderr",
            "error",
            "submitted",
            "state",
            "response_text",
            "duration_s",
            "token_usage",
            "validation_errors",
            "turn_type",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "lm_call_seconds",
            "worker_execute_seconds",
            "submit_payload",
        ],
    )
    def test_field_still_exists(self, field_name):
        names = {f.name for f in dataclasses.fields(TurnRecord)}
        assert field_name in names

    def test_to_dict_round_trip_preserves_all_live_fields(self):
        record = _make_turn(
            response_text="hello",
            token_usage={"prompt_tokens": 10},
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cached_tokens=2,
            reasoning_tokens=3,
            duration_s=0.5,
            validation_errors=["bad"],
            turn_type="normal",
            submit_payload={"answer": "42"},
        )
        d = record.to_dict()
        assert d["response_text"] == "hello"
        assert d["token_usage"] == {"prompt_tokens": 10}
        assert d["prompt_tokens"] == 10
        assert d["completion_tokens"] == 5
        assert d["total_tokens"] == 15
        assert d["cached_tokens"] == 2
        assert d["reasoning_tokens"] == 3
        assert d["duration_s"] == 0.5
        assert d["validation_errors"] == ["bad"]
        assert d["turn_type"] == "normal"
        assert d["submit_payload"] == {"answer": "42"}


class TestBackwardsCompatLoadingOldJsonl:
    """Old JSONL files written before the cleanup contain `"cost": {}`. Replay
    must continue to load them without error — the unknown key is just ignored
    by the canonical-fields lifter in replay._normalize_turn.
    """

    def test_replay_load_tolerates_legacy_cost_key(self, tmp_path):
        from fabric_rlm.replay import load_trajectory

        legacy_jsonl = (
            json.dumps({"metadata": {"created": "2025-01-01"}}) + "\n"
            + json.dumps({
                "turn": 1,
                "code": "print('hi')",
                "stdout": "hi\n",
                "stderr": "",
                "error": None,
                "submitted": False,
                "state": {},
                "cost": {},  # legacy dead field
                "validation_errors": [],
                "turn_type": "normal",
            }) + "\n"
        )
        path = tmp_path / "legacy.jsonl"
        path.write_text(legacy_jsonl, encoding="utf-8")

        loaded = load_trajectory(path)
        assert loaded["metadata"] == {"created": "2025-01-01"}
        assert len(loaded["turns"]) == 1
        assert loaded["turns"][0]["code"] == "print('hi')"
        assert loaded["turns"][0]["turn_type"] == "normal"
        # Cost must NOT appear at the canonical top level — it's a removed
        # field, not a canonical one — even though the raw legacy line had it.
        assert "cost" not in loaded["turns"][0]
        # Legacy "cost" key is tucked under _raw, ignored by canonical view.
        assert loaded["turns"][0]["_raw"]["cost"] == {}
