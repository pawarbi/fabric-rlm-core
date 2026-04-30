"""Opt-in adaptive execution helpers.

The default :class:`fabric_rlm.RLM` path stays sequential. This module provides
validator-driven fallback/fanout for callers that explicitly want it.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class AdaptiveStrategy:
    name: str
    fn: Callable[[], Any]
    cost_hint: float | None = None


@dataclass
class StrategyResult:
    name: str
    value: Any
    passed: bool
    error: str | None = None
    cost: dict[str, Any] | None = None


@dataclass
class AdaptiveRunResult:
    value: Any
    winning_strategy: str
    passed: bool
    fanout_used: bool
    attempts: list[StrategyResult] = field(default_factory=list)


class AdaptiveOrchestrator:
    """Run a cheap strategy first, then fan out backups only if validation fails."""

    def __init__(
        self,
        validator: Callable[[Any], bool],
        *,
        max_workers: int = 4,
    ):
        self.validator = validator
        self.max_workers = max_workers

    def run(
        self,
        primary: AdaptiveStrategy,
        backups: Iterable[AdaptiveStrategy] = (),
    ) -> AdaptiveRunResult:
        attempts: list[StrategyResult] = []
        primary_result = self._run_strategy(primary)
        attempts.append(primary_result)
        if primary_result.passed:
            return AdaptiveRunResult(
                value=primary_result.value,
                winning_strategy=primary_result.name,
                passed=True,
                fanout_used=False,
                attempts=attempts,
            )

        backup_list = list(backups)
        if not backup_list:
            return AdaptiveRunResult(
                value=primary_result.value,
                winning_strategy=primary_result.name,
                passed=False,
                fanout_used=False,
                attempts=attempts,
            )

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(backup_list))) as executor:
            futures = {executor.submit(self._run_strategy, strategy): strategy for strategy in backup_list}
            for future in as_completed(futures):
                result = future.result()
                attempts.append(result)
                if result.passed:
                    return AdaptiveRunResult(
                        value=result.value,
                        winning_strategy=result.name,
                        passed=True,
                        fanout_used=True,
                        attempts=attempts,
                    )

        last = attempts[-1]
        return AdaptiveRunResult(
            value=last.value,
            winning_strategy=last.name,
            passed=False,
            fanout_used=True,
            attempts=attempts,
        )

    def _run_strategy(self, strategy: AdaptiveStrategy) -> StrategyResult:
        try:
            value = strategy.fn()
            return StrategyResult(
                name=strategy.name,
                value=value,
                passed=self.validator(value),
                cost={"cost_hint": strategy.cost_hint},
            )
        except Exception as exc:
            return StrategyResult(
                name=strategy.name,
                value=None,
                passed=False,
                error=repr(exc),
                cost={"cost_hint": strategy.cost_hint},
            )


def first_passing(
    strategies: Iterable[tuple[str, Callable[[], Any]]],
    validator: Callable[[Any], bool],
) -> StrategyResult:
    """Run strategies sequentially and return the first validator-approved result."""

    last: StrategyResult | None = None
    for name, strategy in strategies:
        value = strategy()
        result = StrategyResult(name=name, value=value, passed=validator(value), cost={})
        if result.passed:
            return result
        last = result
    if last is None:
        raise ValueError("No strategies supplied")
    return last


SELF_REPORT_FIELDS: tuple[str, ...] = (
    "confidence",
    "result_is_empty",
    "rows_examined",
    "evidence",
)

SELF_REPORT_INSTRUCTIONS: str = (
    "SELF-REPORT CONTRACT (in addition to the answer):\n"
    "- confidence (float, 0.0..1.0): your honest probability the answer is correct.\n"
    "- result_is_empty (bool): True if the answer is 'no/none/zero/not found'.\n"
    "- rows_examined (int): how many rows/records/events you actually scanned.\n"
    "- evidence (str): a short proof of work — the SQL/code, row counts, source, key column.\n"
    "If result_is_empty=True you MUST set rows_examined > 0 AND put a count proof in evidence.\n"
    "Never claim 'no results' without showing how many rows you searched.\n"
)


def with_self_report(signature: str) -> str:
    """Augment ``inputs -> answer: ...`` with the universal self-report outputs.

    Idempotent: if all self-report fields are already present, the signature is
    returned unchanged. Works for any task — knows nothing about the domain.
    """

    if "->" not in signature:
        raise ValueError(f"signature must contain '->', got: {signature!r}")
    lhs, rhs = signature.split("->", 1)
    rhs = rhs.strip()
    if all(field in rhs for field in SELF_REPORT_FIELDS):
        return signature
    extras = ", ".join(
        [
            "confidence: float",
            "result_is_empty: bool",
            "rows_examined: int",
            "evidence: str",
        ]
    )
    sep = ", " if rhs else ""
    return f"{lhs.strip()} -> {rhs}{sep}{extras}"


def universal_validator(
    *,
    min_confidence: float = 0.7,
    min_rows_when_empty: int = 1,
    min_evidence_chars: int = 20,
    answer_keys: tuple[str, ...] = ("answer", "output", "result", "report"),
) -> Callable[[Any], bool]:
    """A task-agnostic validator built on the self-report contract.

    Accepts either a raw payload mapping or any object with ``.submitted`` and
    ``.payload`` (e.g. :class:`fabric_rlm.RLMResult`).

    Pass criteria (in order):
      1. If the value exposes ``.submitted``, it must be ``True``.
      2. ``payload`` must be a non-empty mapping with at least one of
         ``answer_keys`` present and non-blank.
      3. If ``confidence`` is present, it must be ``>= min_confidence``.
      4. If ``result_is_empty`` is ``True``: ``rows_examined`` must be
         ``>= min_rows_when_empty`` AND ``evidence`` must be a string of
         at least ``min_evidence_chars`` characters.

    Self-report fields that are absent are not checked, so the validator
    remains usable even when callers do not opt into the contract — but
    it is strictly more powerful when paired with :func:`with_self_report`.
    """

    def _is_blank(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str):
            return not v.strip()
        if isinstance(v, (list, tuple, set, frozenset, dict)):
            return len(v) == 0
        return False

    def _validator(value: Any) -> bool:
        if hasattr(value, "submitted") and hasattr(value, "payload"):
            if not getattr(value, "submitted"):
                return False
            payload = getattr(value, "payload")
        elif isinstance(value, Mapping):
            payload = value
        else:
            return False

        if not isinstance(payload, Mapping) or not payload:
            return False

        if not any(k in payload and not _is_blank(payload[k]) for k in answer_keys):
            return False

        if "confidence" in payload:
            try:
                conf = float(payload["confidence"])
            except (TypeError, ValueError):
                return False
            if conf < min_confidence:
                return False

        if bool(payload.get("result_is_empty", False)):
            try:
                rows = int(payload.get("rows_examined", 0))
            except (TypeError, ValueError):
                return False
            if rows < min_rows_when_empty:
                return False
            evidence = payload.get("evidence", "")
            if not isinstance(evidence, str) or len(evidence.strip()) < min_evidence_chars:
                return False

        return True

    return _validator

