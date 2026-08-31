"""Immutable contracts for the experimental deep-insight analysis DAG."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
import math
from types import MappingProxyType
from typing import Literal


ExecutionMode = Literal["sequential", "parallel"]
InterpretationLevel = Literal["descriptive", "associational", "predictive"]
TemporalIntent = Literal[
    "current_state",
    "recent_change",
    "historical_context",
    "structural_pattern",
]
RecencyPolicy = Literal["strict", "allow_historical"]
OperatorStatus = Literal["completed", "failed"]
EvidenceState = Literal[
    "planned",
    "running",
    "completed",
    "failed",
    "superseded",
    "rejected",
]

_EXECUTION_MODES = {"sequential", "parallel"}
_INTERPRETATION_LEVELS = {"descriptive", "associational", "predictive"}
_TEMPORAL_INTENTS = {
    "current_state",
    "recent_change",
    "historical_context",
    "structural_pattern",
}
_RECENCY_POLICIES = {"strict", "allow_historical"}
_OPERATOR_STATUSES = {"completed", "failed"}
_EVIDENCE_STATES = {
    "planned",
    "running",
    "completed",
    "failed",
    "superseded",
    "rejected",
}


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _unique_text_tuple(values: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence of strings")
    normalized = tuple(
        _required_text(value, f"{field_name}[{index}]")
        for index, value in enumerate(values)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


def _nonnegative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: object, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _positive_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


def _freeze_json(value: object, field_name: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be JSON-compatible")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{field_name} must have string object keys")
        # Canonical key order makes equivalent inputs serialize identically.
        for key in sorted(value):
            frozen[key] = _freeze_json(value[key], f"{field_name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValueError(f"{field_name} must be JSON-compatible")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class AnalysisBrief:
    """User intent and interpretation limits for one analysis run."""

    objective: str
    focus_areas: tuple[str, ...] = ()
    target_metrics: tuple[str, ...] = ()
    time_window: str | None = None
    comparison_basis: str | None = None
    temporal_intent: TemporalIntent = "historical_context"
    requested_as_of: str | None = None
    recency_policy: RecencyPolicy = "strict"
    latest_complete_period_only: bool = True
    population: str | None = None
    exclusions: tuple[str, ...] = ()
    privacy_constraints: tuple[str, ...] = ()
    interpretation_level: InterpretationLevel = "descriptive"

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective", _required_text(self.objective, "objective"))
        for field_name in (
            "focus_areas",
            "target_metrics",
            "exclusions",
            "privacy_constraints",
        ):
            object.__setattr__(
                self,
                field_name,
                _unique_text_tuple(getattr(self, field_name), field_name),
            )
        for field_name in ("time_window", "comparison_basis", "population"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _required_text(value, field_name))
        if self.temporal_intent not in _TEMPORAL_INTENTS:
            raise ValueError(
                "temporal_intent must be current_state, recent_change, "
                "historical_context, or structural_pattern"
            )
        if self.requested_as_of is not None:
            requested_as_of = _required_text(
                self.requested_as_of,
                "requested_as_of",
            )
            try:
                date.fromisoformat(requested_as_of)
            except ValueError:
                raise ValueError(
                    "requested_as_of must be an ISO calendar date"
                ) from None
            object.__setattr__(self, "requested_as_of", requested_as_of)
        if self.recency_policy not in _RECENCY_POLICIES:
            raise ValueError(
                "recency_policy must be strict or allow_historical"
            )
        if type(self.latest_complete_period_only) is not bool:
            raise ValueError("latest_complete_period_only must be boolean")
        if self.interpretation_level not in _INTERPRETATION_LEVELS:
            raise ValueError(
                "interpretation_level must be descriptive, associational, or predictive"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "focus_areas": list(self.focus_areas),
            "target_metrics": list(self.target_metrics),
            "time_window": self.time_window,
            "comparison_basis": self.comparison_basis,
            "temporal_intent": self.temporal_intent,
            "requested_as_of": self.requested_as_of,
            "recency_policy": self.recency_policy,
            "latest_complete_period_only": self.latest_complete_period_only,
            "population": self.population,
            "exclusions": list(self.exclusions),
            "privacy_constraints": list(self.privacy_constraints),
            "interpretation_level": self.interpretation_level,
        }


@dataclass(frozen=True)
class RunBudget:
    """Hard resource limits validated before DAG execution."""

    max_nodes: int
    max_parallel_nodes: int = 1
    max_rows_per_node: int = 100_000
    timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_nodes", _positive_int(self.max_nodes, "max_nodes"))
        object.__setattr__(
            self,
            "max_parallel_nodes",
            _positive_int(self.max_parallel_nodes, "max_parallel_nodes"),
        )
        object.__setattr__(
            self,
            "max_rows_per_node",
            _positive_int(self.max_rows_per_node, "max_rows_per_node"),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_number(self.timeout_seconds, "timeout_seconds"),
        )
        if self.max_parallel_nodes > self.max_nodes:
            raise ValueError("max_parallel_nodes must not exceed max_nodes")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_nodes": self.max_nodes,
            "max_parallel_nodes": self.max_parallel_nodes,
            "max_rows_per_node": self.max_rows_per_node,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class OperatorNode:
    """One validated request for a registered deterministic operator."""

    node_id: str
    operator: str
    source_ids: tuple[str, ...]
    seed: int
    parameters: Mapping[str, object] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    execution_mode: ExecutionMode = "sequential"
    question: str | None = None
    columns: tuple[str, ...] = ()
    grain: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _required_text(self.node_id, "node_id"))
        object.__setattr__(self, "operator", _required_text(self.operator, "operator"))
        object.__setattr__(
            self,
            "source_ids",
            _unique_text_tuple(self.source_ids, "source_ids"),
        )
        if not self.source_ids:
            raise ValueError("source_ids must not be empty")
        object.__setattr__(self, "seed", _nonnegative_int(self.seed, "seed"))
        object.__setattr__(
            self,
            "depends_on",
            _unique_text_tuple(self.depends_on, "depends_on"),
        )
        for field_name in ("columns", "grain"):
            object.__setattr__(
                self,
                field_name,
                _unique_text_tuple(getattr(self, field_name), field_name),
            )
        if self.execution_mode not in _EXECUTION_MODES:
            raise ValueError("execution_mode must be sequential or parallel")
        if self.question is not None:
            object.__setattr__(self, "question", _required_text(self.question, "question"))
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a JSON-compatible object")
        object.__setattr__(
            self,
            "parameters",
            _freeze_json(self.parameters, "parameters"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "operator": self.operator,
            "source_ids": list(self.source_ids),
            "seed": self.seed,
            "parameters": _thaw_json(self.parameters),
            "depends_on": list(self.depends_on),
            "execution_mode": self.execution_mode,
            "question": self.question,
            "columns": list(self.columns),
            "grain": list(self.grain),
        }


@dataclass(frozen=True)
class AnalysisDAG:
    """A typed analysis plan before host-side validation and scheduling."""

    dag_id: str
    root_seed: int
    budget: RunBudget
    nodes: tuple[OperatorNode, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dag_id", _required_text(self.dag_id, "dag_id"))
        object.__setattr__(self, "root_seed", _nonnegative_int(self.root_seed, "root_seed"))
        if not isinstance(self.budget, RunBudget):
            raise ValueError("budget must be a RunBudget")
        if not isinstance(self.nodes, (list, tuple)) or not self.nodes:
            raise ValueError("nodes must be a non-empty sequence")
        nodes = tuple(self.nodes)
        if any(not isinstance(node, OperatorNode) for node in nodes):
            raise ValueError("nodes must contain only OperatorNode values")
        node_ids = tuple(node.node_id for node in nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("nodes must have unique node_id values")
        # Aggregate budget and graph checks belong to the Task-3 DAG validator.
        object.__setattr__(self, "nodes", nodes)

    def to_dict(self) -> dict[str, object]:
        return {
            "dag_id": self.dag_id,
            "root_seed": self.root_seed,
            "budget": self.budget.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes],
        }


@dataclass(frozen=True)
class OperatorResult:
    """Structured success or failure emitted by one operator node."""

    node_id: str
    operator: str
    status: OperatorStatus
    seed: int
    sample_size: int
    values: Mapping[str, object] = field(default_factory=dict)
    uncertainty: Mapping[str, object] = field(default_factory=dict)
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    failure_code: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _required_text(self.node_id, "node_id"))
        object.__setattr__(self, "operator", _required_text(self.operator, "operator"))
        if self.status not in _OPERATOR_STATUSES:
            raise ValueError("status must be completed or failed")
        object.__setattr__(self, "seed", _nonnegative_int(self.seed, "seed"))
        object.__setattr__(
            self,
            "sample_size",
            _nonnegative_int(self.sample_size, "sample_size"),
        )
        for field_name in ("values", "uncertainty", "diagnostics"):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{field_name} must be a JSON-compatible object")
            object.__setattr__(self, field_name, _freeze_json(value, field_name))
        object.__setattr__(
            self,
            "limitations",
            _unique_text_tuple(self.limitations, "limitations"),
        )
        if self.status == "completed":
            if not self.values:
                raise ValueError("values must not be empty for a completed result")
            if not self.diagnostics:
                raise ValueError(
                    "diagnostics must not be empty for a completed result"
                )
            if self.failure_code is not None or self.failure_message is not None:
                raise ValueError("completed results must not include failure fields")
        else:
            object.__setattr__(
                self,
                "failure_code",
                _required_text(self.failure_code, "failure_code"),
            )
            object.__setattr__(
                self,
                "failure_message",
                _required_text(self.failure_message, "failure_message"),
            )
            if self.values:
                raise ValueError("failed results must not include values")

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "operator": self.operator,
            "status": self.status,
            "seed": self.seed,
            "sample_size": self.sample_size,
            "values": _thaw_json(self.values),
            "uncertainty": _thaw_json(self.uncertainty),
            "diagnostics": _thaw_json(self.diagnostics),
            "limitations": list(self.limitations),
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
        }


@dataclass(frozen=True)
class EvidenceEntry:
    """One immutable evidence record linked to an analysis node."""

    evidence_id: str
    node_id: str
    state: EvidenceState
    result: OperatorResult | None = None
    supersedes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            _required_text(self.evidence_id, "evidence_id"),
        )
        object.__setattr__(self, "node_id", _required_text(self.node_id, "node_id"))
        if self.state not in _EVIDENCE_STATES:
            raise ValueError(f"invalid evidence state: {self.state}")
        if self.result is not None and self.result.node_id != self.node_id:
            raise ValueError("result.node_id must match evidence node_id")
        if self.state in {"completed", "failed"}:
            if self.result is None or self.result.status != self.state:
                raise ValueError(f"{self.state} evidence requires a matching result")
        elif self.result is not None:
            raise ValueError(f"{self.state} evidence must not include a result")
        if self.supersedes is not None:
            object.__setattr__(
                self,
                "supersedes",
                _required_text(self.supersedes, "supersedes"),
            )
            if self.supersedes == self.evidence_id:
                raise ValueError("supersedes must not reference its own evidence_id")

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "node_id": self.node_id,
            "state": self.state,
            "result": self.result.to_dict() if self.result is not None else None,
            "supersedes": self.supersedes,
        }
