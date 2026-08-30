"""Internal traversal and execution of deep-insight numeric evidence checks."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math
from typing import Any


class DeepInsightAuditError(RuntimeError):
    """A deep-insight audit failed at a specific contract path."""


@dataclass(frozen=True, slots=True)
class AuditCheck:
    """One source-derived numeric check passed to a host executor."""

    path: str
    expected: float
    verification: Mapping[str, Any]
    rounding_abs_tol: float = 0.0


@dataclass(frozen=True, slots=True)
class AuditResult:
    """The verified result of one host check."""

    path: str
    expected: float
    actual: float


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Immutable summary of all executed checks."""

    checks: tuple[AuditResult, ...]

    @property
    def total_checks(self) -> int:
        return len(self.checks)


Executor = Callable[[AuditCheck], object]


def audit_deep_insight(
    payload: Mapping[str, Any],
    executor: Executor,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
) -> AuditReport:
    """Execute every authoritative numeric check in a typed contract payload."""

    if payload.get("contract_version") not in {2, 3}:
        raise DeepInsightAuditError(
            "deep insight host audit requires contract_version 2 or 3"
        )
    if not _valid_tolerance(rel_tol) or not _valid_tolerance(abs_tol):
        raise DeepInsightAuditError("audit tolerances must be finite and non-negative")

    results = []
    for check in _iter_checks(payload):
        try:
            raw_result = executor(check)
        except Exception as exc:
            raise DeepInsightAuditError(
                f"executor failed at {check.path}: {exc}"
            ) from exc
        actual = _stable_numeric(_one_numeric_scalar(raw_result, check.path))
        if not _matches_expected(actual, check, rel_tol, abs_tol):
            raise DeepInsightAuditError(
                f"{check.path}: expected {check.expected!r}, actual {actual!r}"
            )
        results.append(
            AuditResult(
                path=check.path,
                expected=check.expected,
                actual=actual,
            )
        )
    return AuditReport(checks=tuple(results))


def _stable_numeric(value: float) -> float:
    return float(format(value, ".12g"))


def _matches_expected(
    actual: float,
    check: AuditCheck,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    if math.isclose(
        actual,
        check.expected,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    ):
        return True
    if check.rounding_abs_tol <= 0:
        return False
    difference = abs(Decimal(str(actual)) - Decimal(str(check.expected)))
    return difference <= Decimal(str(check.rounding_abs_tol))


def _iter_checks(payload: Mapping[str, Any]) -> Iterator[AuditCheck]:
    insights = payload.get("insights", ())
    for insight_index, insight in enumerate(insights):
        if not isinstance(insight, Mapping):
            continue
        insight_path = f"insights[{insight_index}]"
        metric_spec = insight.get("metric_spec")
        if isinstance(metric_spec, Mapping):
            yield from _metric_components(
                metric_spec,
                f"{insight_path}.metric_spec",
            )
        else:
            yield from _legacy_check(insight, insight_path)

        supporting_claims = insight.get("supporting_claims", ())
        for claim_index, claim in enumerate(supporting_claims):
            if not isinstance(claim, Mapping):
                continue
            claim_path = f"{insight_path}.supporting_claims[{claim_index}]"
            claim_metric_spec = claim.get("metric_spec")
            if isinstance(claim_metric_spec, Mapping):
                yield from _metric_components(claim_metric_spec, f"{claim_path}.metric_spec")
            else:
                yield from _legacy_check(claim, claim_path)

        assessment = insight.get("diagnostic_assessment")
        if isinstance(assessment, Mapping):
            explanations = assessment.get("explanations", ())
            for explanation_index, explanation in enumerate(explanations):
                if (
                    isinstance(explanation, Mapping)
                    and explanation.get("disposition")
                    in {"ruled_out", "weakened", "supported"}
                ):
                    path = (
                        f"{insight_path}.diagnostic_assessment."
                        f"explanations[{explanation_index}]"
                    )
                    yield _check(
                        path,
                        explanation.get("expected_value"),
                        explanation.get("verification"),
                    )

    candidates = payload.get("candidates", ())
    for candidate_index, candidate in enumerate(candidates):
        if (
            not isinstance(candidate, Mapping)
            or candidate.get("rejection_type") != "quantitative"
        ):
            continue
        evidence = candidate.get("rejection_evidence")
        if not isinstance(evidence, Mapping):
            continue
        verification = evidence.get("verification")
        if not isinstance(verification, Mapping):
            continue
        components = verification.get("components", ())
        for component_index, component in enumerate(components):
            if not isinstance(component, Mapping):
                continue
            path = (
                f"candidates[{candidate_index}].rejection_evidence."
                f"verification.components[{component_index}]"
            )
            component_verification = component.get("verification")
            if not isinstance(component_verification, Mapping):
                component_verification = {
                    "method": verification.get("method"),
                    "expression": component.get("expression"),
                    "sources": component.get("sources"),
                }
            yield _check(
                path,
                component.get("expected_value"),
                component_verification,
            )


def _metric_components(
    metric_spec: Mapping[str, Any],
    path: str,
) -> Iterator[AuditCheck]:
    for component_index, component in enumerate(metric_spec.get("components", ())):
        if not isinstance(component, Mapping):
            continue
        yield _check(
            f"{path}.components[{component_index}]",
            component.get("expected_value"),
            component.get("verification"),
        )


def _legacy_check(
    item: Mapping[str, Any],
    path: str,
) -> Iterator[AuditCheck]:
    if "expected_value" in item:
        expected = item["expected_value"]
    elif "current_value" in item:
        expected = item["current_value"]
    else:
        return
    yield _check(path, expected, item.get("verification"))


def _check(path: str, expected: object, verification: object) -> AuditCheck:
    expected_number = _numeric(expected)
    if expected_number is None:
        raise DeepInsightAuditError(f"{path}: expected_value must be finite numeric")
    if not isinstance(verification, Mapping):
        raise DeepInsightAuditError(f"{path}: verification must be structured")
    return AuditCheck(
        path=path,
        expected=expected_number,
        verification=verification,
        rounding_abs_tol=_implied_rounding_tolerance(expected),
    )


def _implied_rounding_tolerance(value: object) -> float:
    if type(value) is int:
        return 0.0
    try:
        decimal_value = Decimal(str(value))
    except (ValueError, TypeError):
        return 0.0
    exponent = decimal_value.as_tuple().exponent
    if exponent >= 0:
        return 0.0
    return float(Decimal("0.5") * (Decimal(10) ** exponent))


def _one_numeric_scalar(result: object, path: str) -> float:
    value = result
    for _ in range(8):
        if isinstance(value, Mapping):
            if len(value) != 1:
                break
            value = next(iter(value.values()))
            continue
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            if len(value) != 1:
                break
            value = value[0]
            continue
        number = _numeric(value)
        if number is not None:
            return number
        break
    raise DeepInsightAuditError(
        f"{path}: executor must return exactly one finite numeric scalar"
    )


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, Decimal, Fraction)
    ):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_tolerance(value: object) -> bool:
    number = _numeric(value)
    return number is not None and number >= 0
