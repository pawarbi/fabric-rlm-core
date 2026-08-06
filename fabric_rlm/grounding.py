"""Is the answer derived from what the run actually saw?

A model can put a number in its answer that no cell ever produced. It is not
lying so much as filling a gap: the figure is plausible, the report wants one,
and by the time it writes the answer the work is fifteen turns back.

Observed in a leaderboard submission, and confirmed by the benchmark's own
audit: four trials reported an exact ground-truth value having either never
queried the database or never derived the figure they reported. Three of them
submitted on their first turn, in two milliseconds. The same failure showed up
in report writing as an invented Cpk, a shift comparison that was never run,
and a sigma level of 5.57 for a defect rate that gives 3.49.

Both checks here read the trajectory after the fact. They need no contract with
the model, no cooperation, and they work on runs that have already finished -
including ones recorded before this module existed.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

# Below this, a "figure" is a loop index, a small count, a threshold or a year,
# and chasing it produces noise rather than findings.
_TRIVIAL = 1.0


def numbers_in(text: Any, *, ndigits: int = 4) -> set[float]:
    """Every number in a piece of text, rounded so 3.14159 and 3.1416 agree."""
    found: set[float] = set()
    for match in _NUMBER.finditer(str(text).replace(",", "")):
        try:
            found.add(round(float(match.group(0)), ndigits))
        except ValueError:
            continue
    return found


def observed_numbers(turns: Iterable[Any]) -> set[float]:
    """Numbers the run actually saw: anything a cell printed or errored with."""
    seen: set[float] = set()
    for turn in turns:
        seen |= numbers_in(getattr(turn, "stdout", "") or "")
        seen |= numbers_in(getattr(turn, "stderr", "") or "")
    return seen


def _with_simple_arithmetic(values: set[float], *, ndigits: int = 4) -> set[float]:
    """Values plus the ratios and sums a report may legitimately state.

    An average is a/b and a share is a/b, and neither is usually printed on its
    own before being written down. Without this the check flags ordinary
    arithmetic; with it, the check is looser than it looks - a few dozen
    observed numbers generate thousands of candidates, so treat a pass as weak
    evidence and a flag as worth reading.
    """
    out = set(values)
    for a in values:
        for b in values:
            if b:
                out.add(round(a / b, ndigits))
            out.add(round(a - b, ndigits))
            out.add(round(a + b, ndigits))
    return out


def ungrounded_figures(
    turns: Iterable[Any],
    payload: Any,
    *,
    allow_arithmetic: bool = True,
    ignore_below: float = _TRIVIAL,
) -> list[float]:
    """Figures in the answer that the run never saw and cannot have derived.

    A screen, not a verdict. On real traces it correctly flagged every trial an
    independent audit had voided, and also flagged three multi-turn runs whose
    figures were ratios this reconstruction does not reach. Read what it
    returns; do not gate on it.
    """
    turns = list(turns)
    seen = observed_numbers(turns)
    allowed = _with_simple_arithmetic(seen) if allow_arithmetic else seen
    claimed = numbers_in(payload if isinstance(payload, str) else _payload_text(payload))
    return sorted(
        c for c in claimed
        if abs(c) > ignore_below and c not in allowed and c not in _years()
    )


def _years() -> set[float]:
    return {float(y) for y in range(1900, 2101)}


def _payload_text(payload: Any) -> str:
    if isinstance(payload, dict):
        return " ".join(str(v) for v in payload.values())
    return str(payload)


def submitted_without_evidence(turns: Iterable[Any]) -> bool:
    """True when the run answered having produced no output at all.

    Unlike `ungrounded_figures` this is not a heuristic. A run that submits
    without any cell having printed anything cannot have derived its answer
    from the data, whatever the answer says. It caught every trial in the
    audited submission that was voided for exactly that.
    """
    produced_output = False
    for turn in turns:
        if (getattr(turn, "stdout", "") or "").strip():
            produced_output = True
        if (getattr(turn, "stderr", "") or "").strip():
            produced_output = True
    return not produced_output


def evidence_report(turns: Iterable[Any], payload: Any) -> dict[str, Any]:
    """Both checks together, shaped for a run report."""
    turns = list(turns)
    ungrounded = ungrounded_figures(turns, payload)
    return {
        "observed_figures": len(observed_numbers(turns)),
        "ungrounded_figures": ungrounded,
        "submitted_without_evidence": submitted_without_evidence(turns),
    }
