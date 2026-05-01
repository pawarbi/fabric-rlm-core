"""Offline analysis helpers for adaptive attempt logs.

These functions answer one limited question cheaply: *would another difficulty
signal have escalated at the same point as the one that was actually used?*

They do **not** answer "would another signal have produced a better outcome" —
that requires running the alternative attempts end-to-end (see the bench's
lattice-eval mode). The honest framing is preserved in the function names so
the limitation is visible at the call site.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .adaptive_policy import (
    AttemptRecord,
    DifficultySignal,
    DifficultyVerdict,
)


@dataclass(frozen=True)
class EscalationReplay:
    """Result of replaying a single attempt log against an alternative signal."""

    signal_name: str
    actual_attempts: int
    actual_passed: bool
    would_have_escalated_at: list[int]
    """Indices of attempts where the alternative signal said 'escalate'."""

    matches_actual: bool
    """True iff the alternative signal escalates at the same indices as the
    actual run did. False means the alternative would have made a different
    routing decision at some point — but we cannot know whether the resulting
    attempts would have passed or failed."""


def would_signal_escalate(
    records: Sequence[AttemptRecord],
    signal: DifficultySignal,
) -> EscalationReplay:
    """Replay an attempt log against ``signal`` and report escalation timing.

    Limitations: only inspects whether ``signal.assess(...)`` would have
    returned ``EscalateRung`` at each step. Cannot project whether different
    attempts would have passed.
    """
    name = type(signal).__name__
    actual_passed = bool(records and records[-1].verdict.passed)
    would_escalate_at: list[int] = []
    actual_escalations: list[int] = []

    for i, rec in enumerate(records):
        if i + 1 < len(records) and records[i + 1].rung > rec.rung:
            actual_escalations.append(i)

    for i, rec in enumerate(records):
        prefix = list(records[: i + 1])
        try:
            verdict = signal.assess(prefix, max_rung=4)
        except Exception:
            continue
        if isinstance(verdict, DifficultyVerdict) and verdict.action == "escalate":
            would_escalate_at.append(i)

    return EscalationReplay(
        signal_name=name,
        actual_attempts=len(records),
        actual_passed=actual_passed,
        would_have_escalated_at=would_escalate_at,
        matches_actual=would_escalate_at == actual_escalations,
    )


def compare_escalation_timing(
    case_logs: Iterable[Sequence[AttemptRecord]],
    signals: Sequence[DifficultySignal],
) -> dict[str, dict[str, Any]]:
    """Aggregate :func:`would_signal_escalate` over many cases × many signals.

    Returns a table-like dict keyed by signal name with totals:

        {
          "ValidatorOnly": {"agreement": 18, "disagreement": 2, "cases": 20},
          "Confidence":    {"agreement": 14, "disagreement": 6, "cases": 20},
        }

    "agreement" = number of cases where the signal would have escalated at the
    same indices as the actual run; "disagreement" = the rest. This is a
    timing-agreement metric only; it does not measure accuracy.
    """
    out: dict[str, dict[str, Any]] = {
        type(s).__name__: {"agreement": 0, "disagreement": 0, "cases": 0}
        for s in signals
    }
    for log in case_logs:
        for sig in signals:
            replay = would_signal_escalate(log, sig)
            row = out[replay.signal_name]
            row["cases"] += 1
            if replay.matches_actual:
                row["agreement"] += 1
            else:
                row["disagreement"] += 1
    return out


__all__ = [
    "EscalationReplay",
    "would_signal_escalate",
    "compare_escalation_timing",
]
