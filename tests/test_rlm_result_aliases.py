"""Tests for RLMResult convenience aliases (LIB-NEW-4).

Pins three properties (``turns``, ``n_turns``, ``outputs``) plus the v6.5
payload-key fall-through behavior preserved by ``__getattr__``.
"""

from __future__ import annotations

from typing import Any

import pytest

from fabric_rlm.runtime import RLMResult
from fabric_rlm.trajectory import Trajectory, TurnRecord


def _make_turn(idx: int) -> TurnRecord:
    """Build a minimal TurnRecord. Tests don't care about most fields."""
    return TurnRecord(
        turn=idx,
        code="",
        stdout="",
        stderr="",
        error=None,
        submitted=False,
        state={},
    )


def _make_result(
    *,
    payload: dict[str, Any] | None = None,
    n_turns: int = 0,
    submitted: bool | None = None,
) -> RLMResult:
    traj = Trajectory()
    for i in range(n_turns):
        traj.append(_make_turn(i))
    if submitted is None:
        submitted = payload is not None
    return RLMResult(
        submitted=submitted,
        payload=payload,
        trajectory=traj,
        final_state={},
    )


# ---- .turns -----------------------------------------------------------------


def test_turns_returns_trajectory_turns() -> None:
    result = _make_result(n_turns=3)
    assert result.turns is result.trajectory.turns
    assert len(result.turns) == 3


def test_turns_empty_for_zero_turn_run() -> None:
    result = _make_result(n_turns=0)
    assert result.turns == []


# ---- .n_turns ---------------------------------------------------------------


def test_n_turns_matches_len_of_turns() -> None:
    result = _make_result(n_turns=5)
    assert result.n_turns == 5
    assert result.n_turns == len(result.trajectory.turns)


def test_n_turns_zero_when_empty() -> None:
    result = _make_result(n_turns=0)
    assert result.n_turns == 0


# ---- .outputs ---------------------------------------------------------------


def test_outputs_returns_payload_when_submitted() -> None:
    result = _make_result(payload={"answer": 42, "confidence": 0.9})
    assert result.outputs == {"answer": 42, "confidence": 0.9}
    assert result.outputs.get("answer") == 42


def test_outputs_returns_empty_dict_when_no_submission() -> None:
    """``.outputs.get(...)`` must be safe on no-submit runs."""
    result = _make_result(payload=None, submitted=False)
    assert result.outputs == {}
    assert result.outputs.get("answer") is None


# ---- v6.5 payload-key fall-through (must still work) ------------------------


def test_payload_keys_still_resolve_via_getattr() -> None:
    """Non-reserved payload keys remain accessible via attribute syntax."""
    result = _make_result(payload={"answer": 7, "explanation": "because"})
    assert result.answer == 7
    assert result.explanation == "because"


def test_property_wins_over_payload_key_collision() -> None:
    """If payload contains a key that collides with a property name,
    the property descriptor wins (Python attribute-lookup order). This is
    the documented contract: callers requesting reserved names see the
    property, never the colliding payload value.
    """
    bogus_outputs = {"this": "should not be returned"}
    result = _make_result(payload={"outputs": bogus_outputs, "answer": 1})
    # property returns the whole payload, not payload["outputs"]
    assert result.outputs is not bogus_outputs
    assert result.outputs == {"outputs": bogus_outputs, "answer": 1}


@pytest.mark.parametrize("reserved_name", ["turns", "n_turns", "outputs"])
def test_all_reserved_names_shadow_payload_keys(reserved_name: str) -> None:
    """All three reserved names (``turns``, ``n_turns``, ``outputs``) shadow
    same-named payload keys. Pinning this for every reserved name (not just
    ``outputs``) ensures the contract is uniform: callers needing the payload
    value of a colliding key must use ``result.payload[name]``.
    """
    sentinel = "should-not-be-returned"
    result = _make_result(payload={reserved_name: sentinel}, n_turns=2)
    value = getattr(result, reserved_name)
    assert value != sentinel, (
        f"Property {reserved_name!r} should shadow payload key, "
        f"but got the payload value back."
    )


def test_missing_attribute_still_raises() -> None:
    result = _make_result(payload={"answer": 1})
    with pytest.raises(AttributeError):
        _ = result.nonexistent_field
