"""Host-side validation and scheduling for experimental analysis DAGs."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass

from fabric_rlm.experimental.analysis_contracts import AnalysisDAG, OperatorNode


def _text_tuple(values: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence of strings")
    normalized: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name}[{index}] must be a non-empty string")
        normalized.append(value.strip())
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return tuple(normalized)


@dataclass(frozen=True)
class OperatorSpec:
    """Parameters accepted by one registered deterministic operator."""

    operator: str
    allowed_parameters: tuple[str, ...] = ()
    required_parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.operator, str) or not self.operator.strip():
            raise ValueError("operator must be a non-empty string")
        object.__setattr__(self, "operator", self.operator.strip())
        object.__setattr__(
            self,
            "allowed_parameters",
            _text_tuple(self.allowed_parameters, "allowed_parameters"),
        )
        object.__setattr__(
            self,
            "required_parameters",
            _text_tuple(self.required_parameters, "required_parameters"),
        )
        unexpected = sorted(
            set(self.required_parameters) - set(self.allowed_parameters)
        )
        if unexpected:
            raise ValueError(
                "required_parameters must also be allowed_parameters: "
                + ", ".join(unexpected)
            )


@dataclass(frozen=True)
class ValidatedDAG:
    """A validated DAG and its deterministic execution waves."""

    dag: AnalysisDAG
    waves: tuple[tuple[str, ...], ...]

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(node_id for wave in self.waves for node_id in wave)


def _validate_node(
    node: OperatorNode,
    *,
    nodes_by_id: Mapping[str, OperatorNode],
    operator_registry: Mapping[str, OperatorSpec],
    authorized_sources: Set[str],
) -> None:
    spec = operator_registry.get(node.operator)
    if spec is None:
        raise ValueError(
            f"node {node.node_id} operator is not registered: {node.operator}"
        )
    if spec.operator != node.operator:
        raise ValueError(
            f"node {node.node_id} operator registry key '{node.operator}' "
            f"does not match spec operator '{spec.operator}'"
        )

    unauthorized = sorted(set(node.source_ids) - set(authorized_sources))
    if unauthorized:
        raise ValueError(
            f"node {node.node_id} source_ids are unauthorized: "
            + ", ".join(unauthorized)
        )

    missing_dependencies = sorted(set(node.depends_on) - nodes_by_id.keys())
    if missing_dependencies:
        raise ValueError(
            f"node {node.node_id} depends_on missing nodes: "
            + ", ".join(missing_dependencies)
        )

    supplied = set(node.parameters)
    unsupported = sorted(supplied - set(spec.allowed_parameters))
    if unsupported:
        raise ValueError(
            f"node {node.node_id} parameters.{unsupported[0]} is not allowed"
        )
    missing_parameters = sorted(set(spec.required_parameters) - supplied)
    if missing_parameters:
        raise ValueError(
            f"node {node.node_id} parameters.{missing_parameters[0]} is required"
        )


def _build_waves(
    dag: AnalysisDAG,
    nodes_by_id: Mapping[str, OperatorNode],
) -> tuple[tuple[str, ...], ...]:
    remaining_dependencies = {
        node.node_id: set(node.depends_on) for node in dag.nodes
    }
    dependents: dict[str, set[str]] = {node_id: set() for node_id in nodes_by_id}
    for node in dag.nodes:
        for dependency in node.depends_on:
            dependents[dependency].add(node.node_id)

    waves: list[tuple[str, ...]] = []
    completed: set[str] = set()
    while len(completed) < len(nodes_by_id):
        ready = sorted(
            node_id
            for node_id, dependencies in remaining_dependencies.items()
            if node_id not in completed and not dependencies
        )
        if not ready:
            unresolved = ", ".join(sorted(set(nodes_by_id) - completed))
            raise ValueError(f"analysis DAG contains a cycle involving: {unresolved}")

        sequential = [
            node_id
            for node_id in ready
            if nodes_by_id[node_id].execution_mode == "sequential"
        ]
        if sequential:
            # Sequential nodes run alone in stable lexical order.
            wave = (sequential[0],)
        else:
            wave = tuple(ready[: dag.budget.max_parallel_nodes])
        waves.append(wave)

        for node_id in wave:
            completed.add(node_id)
            for dependent in dependents[node_id]:
                remaining_dependencies[dependent].discard(node_id)

    return tuple(waves)


def validate_analysis_dag(
    dag: AnalysisDAG,
    *,
    operator_registry: Mapping[str, OperatorSpec],
    authorized_sources: Set[str],
) -> ValidatedDAG:
    """Reject unsafe plans and return a stable bounded execution schedule."""

    if len(dag.nodes) > dag.budget.max_nodes:
        raise ValueError(
            f"analysis DAG has {len(dag.nodes)} nodes, exceeding max_nodes "
            f"{dag.budget.max_nodes}"
        )
    if not isinstance(operator_registry, Mapping):
        raise ValueError("operator_registry must be a mapping")
    if not isinstance(authorized_sources, Set):
        raise ValueError("authorized_sources must be a set")

    nodes_by_id = {node.node_id: node for node in dag.nodes}
    for node in dag.nodes:
        _validate_node(
            node,
            nodes_by_id=nodes_by_id,
            operator_registry=operator_registry,
            authorized_sources=authorized_sources,
        )

    return ValidatedDAG(dag=dag, waves=_build_waves(dag, nodes_by_id))
