"""Current-source drift checks for immutable knowledge packages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

from fabric_rlm.knowledge import (
    KnowledgeEvent,
    KnowledgePackage,
    RegisteredOperation,
    Relationship,
    SourceProfile,
    SourceRole,
    Status,
    SubjectType,
    _domain_fingerprint,
)
from fabric_rlm.knowledge_sources import (
    ProfileLimits,
    SourceAdapterRegistry,
    profile_sources,
)


DriftKind = Literal["snapshot", "schema", "inexact"]


@dataclass(frozen=True)
class KnowledgePreflightResult:
    """A package lifecycle result plus separately observed current profiles."""

    package: KnowledgePackage
    current_profiles: Mapping[str, SourceProfile]
    drift: Mapping[str, DriftKind]

    def __post_init__(self) -> None:
        if not isinstance(self.package, KnowledgePackage):
            raise ValueError("package must be a KnowledgePackage")
        object.__setattr__(
            self,
            "current_profiles",
            MappingProxyType(dict(sorted(self.current_profiles.items()))),
        )
        object.__setattr__(
            self,
            "drift",
            MappingProxyType(dict(sorted(self.drift.items()))),
        )

    @property
    def is_current(self) -> bool:
        return not self.drift


def _exact_aliases(
    package: KnowledgePackage,
    sources: Mapping[str, object],
) -> None:
    expected = {source.source_id for source in package.sources}
    actual = set(sources)
    missing = sorted(expected - actual)
    extras = sorted(actual - expected)
    if missing or extras:
        details: list[str] = []
        if missing:
            details.append(f"missing aliases: {', '.join(missing)}")
        if extras:
            details.append(f"extra aliases: {', '.join(extras)}")
        raise ValueError("sources must use exact aliases; " + "; ".join(details))


def _persisted_roles(package: KnowledgePackage) -> dict[str, SourceRole]:
    return {source.source_id: source.role for source in package.sources}


def _validated_roles(
    package: KnowledgePackage,
    roles: Mapping[str, object] | None,
) -> dict[str, SourceRole]:
    persisted = _persisted_roles(package)
    if roles is None:
        return persisted
    if not isinstance(roles, Mapping):
        raise TypeError("roles must be a mapping")
    if set(roles) != set(persisted):
        raise ValueError("role overrides must use exact source aliases")
    mismatched = sorted(
        source_id
        for source_id, role in roles.items()
        if role != persisted[source_id]
    )
    if mismatched:
        raise ValueError(f"role mismatch for source alias: {mismatched[0]}")
    return persisted


def _reason_code(
    source_ids: set[str],
    drift: Mapping[str, DriftKind],
) -> str:
    if any(drift[source_id] == "schema" for source_id in source_ids):
        return "source_schema_drift"
    if any(drift[source_id] == "inexact" for source_id in source_ids):
        return "source_snapshot_inexact"
    return "source_snapshot_drift"


def _stale_status(status: Status) -> Status:
    return status if status in {"quarantined", "retired"} else "stale"


def _event(
    *,
    subject_type: SubjectType,
    subject_id: str,
    reason_code: str,
) -> KnowledgeEvent:
    suffix = _domain_fingerprint(
        "knowledge-preflight-event-v1",
        {
            "subject_type": subject_type,
            "subject_id": subject_id,
            "reason_code": reason_code,
        },
    )[:16]
    return KnowledgeEvent(
        event_id=f"preflight.{subject_type}.{subject_id}.{suffix}",
        event_type=f"{subject_type}.stale",
        subject_type=subject_type,
        subject_id=subject_id,
        status="stale",
        reason_code=reason_code,
    )


def preflight_knowledge(
    package: KnowledgePackage,
    sources: Mapping[str, object],
    *,
    roles: Mapping[str, object] | None = None,
    limits: ProfileLimits | None = None,
    registry: SourceAdapterRegistry | None = None,
) -> KnowledgePreflightResult:
    """Reprofile exact bindings and stale source-bound knowledge on drift."""

    if not isinstance(package, KnowledgePackage):
        raise TypeError("package must be a KnowledgePackage")
    if not isinstance(sources, Mapping):
        raise TypeError("sources must be a mapping")
    _exact_aliases(package, sources)
    active_roles = _validated_roles(package, roles)
    current = profile_sources(
        sources,
        roles=active_roles,
        limits=limits,
        registry=registry,
    )
    current_by_id = {source.source_id: source for source in current}
    learned_by_id = {source.source_id: source for source in package.sources}
    drift: dict[str, DriftKind] = {}
    for source_id in sorted(learned_by_id):
        learned = learned_by_id[source_id]
        observed = current_by_id[source_id]
        if observed.diagnostics.get("snapshot_exact") is not True:
            drift[source_id] = "inexact"
        elif learned.schema_fingerprint != observed.schema_fingerprint:
            drift[source_id] = "schema"
        elif learned.snapshot_fingerprint != observed.snapshot_fingerprint:
            drift[source_id] = "snapshot"

    if not drift:
        return KnowledgePreflightResult(
            package=package,
            current_profiles=current_by_id,
            drift={},
        )

    changed_ids = set(drift)
    sources_out: list[SourceProfile] = []
    events: list[KnowledgeEvent] = list(package.events)
    existing_event_ids = {event.event_id for event in events}

    def add_event(event: KnowledgeEvent) -> None:
        if event.event_id not in existing_event_ids:
            events.append(event)
            existing_event_ids.add(event.event_id)

    for source in package.sources:
        if source.source_id not in changed_ids:
            sources_out.append(source)
            continue
        updated = replace(source, status=_stale_status(source.status))
        sources_out.append(updated)
        if updated.status == "stale":
            add_event(
                _event(
                    subject_type="source",
                    subject_id=source.source_id,
                    reason_code=_reason_code({source.source_id}, drift),
                )
            )

    relationships_out: list[Relationship] = []
    stale_relationship_ids: set[str] = set()
    for relationship in package.relationships:
        dependencies = {
            relationship.left_source,
            relationship.right_source,
        } & changed_ids
        if not dependencies:
            relationships_out.append(relationship)
            continue
        reason_code = _reason_code(dependencies, drift)
        updated = (
            relationship
            if relationship.status in {"quarantined", "retired"}
            else replace(
                relationship,
                status="stale",
                reason_code=reason_code,
            )
        )
        relationships_out.append(updated)
        stale_relationship_ids.add(relationship.relationship_id)
        if updated.status == "stale":
            add_event(
                _event(
                    subject_type="relationship",
                    subject_id=relationship.relationship_id,
                    reason_code=reason_code,
                )
            )

    operations_out: list[RegisteredOperation] = []
    for operation in package.operations:
        source_dependencies = set(operation.required_sources) & changed_ids
        relationship_dependencies = (
            set(operation.required_relationships) & stale_relationship_ids
        )
        if not source_dependencies and not relationship_dependencies:
            operations_out.append(operation)
            continue
        related_sources = set(source_dependencies)
        for relationship in package.relationships:
            if relationship.relationship_id in relationship_dependencies:
                related_sources.update(
                    {relationship.left_source, relationship.right_source}
                    & changed_ids
                )
        reason_code = _reason_code(related_sources, drift)
        updated = (
            operation
            if operation.status in {"quarantined", "retired"}
            else replace(
                operation,
                status="stale",
                reason_code=reason_code,
            )
        )
        operations_out.append(updated)
        if updated.status == "stale":
            add_event(
                _event(
                    subject_type="operation",
                    subject_id=operation.operation_id,
                    reason_code=reason_code,
                )
            )

    updated_package = KnowledgePackage(
        package_id=package.package_id,
        sources=tuple(sources_out),
        relationships=tuple(relationships_out),
        operations=tuple(operations_out),
        events=tuple(events),
    )
    return KnowledgePreflightResult(
        package=updated_package,
        current_profiles=current_by_id,
        drift=drift,
    )


__all__: list[str] = []
