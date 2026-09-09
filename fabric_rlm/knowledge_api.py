"""Public orchestration for learning and rebinding knowledge packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import os

from dataclasses import replace

from fabric_rlm.knowledge import (
    EvidenceRecord,
    KnowledgeEvent,
    KnowledgePackage,
    SourceProfile,
    SourceRole,
    _domain_fingerprint,
)
from fabric_rlm.knowledge_evidence import harvest_evidence, run_fingerprint_for
from fabric_rlm.knowledge_lakehouse_sources import fabric_source_registry
from fabric_rlm.knowledge_lessons import (
    current_evidence,
    declared_lessons,
    promote_lessons,
    structural_lessons,
)
from fabric_rlm.knowledge_operations import discover_registered_operations
from fabric_rlm.knowledge_sources import (
    ProfileLimits,
    SourceAdapterRegistry,
    profile_sources,
)
from fabric_rlm.knowledge_store import (
    BoundKnowledgePackage,
    SourceBinding,
    SourceBindingDescriptor,
    bind_knowledge_package,
    read_knowledge_package,
    save_knowledge_package,
)
from fabric_rlm.onelake_knowledge_store import (
    OneLakeKnowledgeLocation,
    OneLakeKnowledgeTransport,
    OneLakeRestTransport,
    read_onelake_knowledge_package,
    save_onelake_knowledge_package,
)


KnowledgeStore = str | os.PathLike[str] | OneLakeKnowledgeLocation


@dataclass(frozen=True)
class Knowledge(BoundKnowledgePackage):
    """A package plus opaque bindings and runtime-only profiling context."""

    _registry: SourceAdapterRegistry = field(
        default_factory=fabric_source_registry,
        repr=False,
        compare=False,
    )
    _limits: ProfileLimits = field(
        default_factory=ProfileLimits,
        repr=False,
        compare=False,
    )


def _active_registry(
    registry: SourceAdapterRegistry | None,
) -> SourceAdapterRegistry:
    from fabric_rlm.knowledge_sources import _validated_registry

    candidate = fabric_source_registry() if registry is None else registry
    return _validated_registry(candidate)


def _package_id(
    profiles: tuple[SourceProfile, ...],
    package_id: str | None,
) -> str:
    if package_id is not None:
        return package_id
    structural_identity = [
        {
            "source_id": profile.source_id,
            "family": profile.family,
            "role": profile.role,
        }
        for profile in profiles
    ]
    suffix = _domain_fingerprint(
        "fabric-rlm.knowledge.package-id.v1",
        structural_identity,
    )[:20]
    return f"knowledge.{suffix}"


def _bindings_from_profiles(
    profiles: tuple[SourceProfile, ...],
    sources: Mapping[str, object],
) -> dict[str, SourceBinding]:
    return {
        profile.source_id: SourceBinding(
            descriptor=SourceBindingDescriptor(
                source_id=profile.source_id,
                locator=profile.locator,
            ),
            value=sources[profile.source_id],
        )
        for profile in profiles
    }


def _exact_aliases(
    package: KnowledgePackage,
    sources: Mapping[str, object],
) -> None:
    expected = {profile.source_id for profile in package.sources}
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


def _load_roles(
    package: KnowledgePackage,
    roles: Mapping[str, object] | None,
) -> dict[str, SourceRole]:
    persisted = {profile.source_id: profile.role for profile in package.sources}
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


def _validate_current_profiles(
    package: KnowledgePackage,
    profiles: tuple[SourceProfile, ...],
) -> None:
    persisted = {profile.source_id: profile for profile in package.sources}
    drift: list[str] = []
    for current in profiles:
        learned = persisted[current.source_id]
        if (
            current.diagnostics.get("snapshot_exact") is not True
            or current.schema_fingerprint != learned.schema_fingerprint
            or current.snapshot_fingerprint != learned.snapshot_fingerprint
        ):
            drift.append(current.source_id)
    if drift:
        raise ValueError(
            "stale knowledge sources detected while loading: "
            + ", ".join(sorted(drift))
        )


def _onelake_location(store: KnowledgeStore) -> OneLakeKnowledgeLocation | None:
    if isinstance(store, OneLakeKnowledgeLocation):
        return store
    if not isinstance(store, str) or not store.lower().startswith("abfss://"):
        return None
    prefix, separator, locator = store.partition("/Files/")
    if not separator or not locator:
        raise ValueError(
            "ABFSS knowledge store must identify a file below a OneLake Files root"
        )
    return OneLakeKnowledgeLocation(
        root=f"{prefix}/Files",
        locator=locator,
    )


def learn(
    *,
    sources: Mapping[str, object],
    store: KnowledgeStore | None = None,
    roles: Mapping[str, object] | None = None,
    package_id: str | None = None,
    limits: ProfileLimits | None = None,
    registry: SourceAdapterRegistry | None = None,
    transport: OneLakeKnowledgeTransport | None = None,
    overwrite: bool = False,
    declared: Mapping[str, Mapping[str, object]] | None = None,
) -> Knowledge:
    """Profile approved sources and return an immutable, runtime-bound package.

    ``declared`` carries what the source owner knows and the profile cannot
    infer: the grain, the period column, units and definitions (see
    :func:`fabric_rlm.knowledge_lessons.declared_lessons`). Declared facts
    become active lessons that reach every task on the source.
    """

    active_limits = limits or ProfileLimits()
    active_registry = _active_registry(registry)
    profiles = profile_sources(
        sources,
        roles=roles,
        limits=active_limits,
        registry=active_registry,
    )
    package = KnowledgePackage(
        package_id=_package_id(profiles, package_id),
        sources=profiles,
        operations=discover_registered_operations(profiles, sources),
    )
    # Structural lessons: what the sources declare about themselves (a
    # semantic model's current-period construct). Sources that declare
    # nothing produce no lessons, and the package is then exactly what it
    # was before learning existed.
    lessons = {lesson.lesson_id: lesson for lesson in structural_lessons(package)}
    for lesson in declared_lessons(package, declared):
        lessons[lesson.lesson_id] = lesson
    package = KnowledgePackage(
        package_id=package.package_id,
        sources=package.sources,
        relationships=package.relationships,
        operations=package.operations,
        events=package.events,
        lessons=tuple(lessons[key] for key in sorted(lessons)),
    )
    bindings = _bindings_from_profiles(profiles, sources)
    if store is not None:
        _persist(store, package, transport=transport, overwrite=overwrite)
    return Knowledge(
        package=package,
        bindings={alias: binding.value for alias, binding in bindings.items()},
        _registry=active_registry,
        _limits=active_limits,
    )


def _observed_fingerprints(result: object, alias_map: Mapping[str, str]) -> dict[str, str]:
    """Schema fingerprints the run executed against, keyed by package source id.

    The runtime records them on the trajectory when a package is bound;
    evidence harvested at run time carries the same stamp. Either is the
    execution-time identity. An empty mapping means the run recorded none.
    """
    metadata = getattr(getattr(result, "trajectory", None), "metadata", None) or {}
    recorded = metadata.get("knowledge_source_fingerprints")
    observed: dict[str, str] = {}
    if isinstance(recorded, Mapping):
        for source_id, fingerprint in recorded.items():
            if isinstance(source_id, str) and isinstance(fingerprint, str) and fingerprint:
                observed[alias_map.get(source_id, source_id)] = fingerprint
    if observed:
        return observed
    for record in getattr(result, "evidence", ()) or ():
        if not isinstance(record, EvidenceRecord):
            continue
        for source_id, fingerprint in record.source_fingerprints.items():
            mapped = alias_map.get(source_id, source_id)
            if observed.get(mapped, fingerprint) != fingerprint:
                return {}  # one run, two identities for a source: unusable
            observed[mapped] = fingerprint
    return observed


def _reusable_evidence(result: object, observed: Mapping[str, str]) -> list[EvidenceRecord]:
    """``result.evidence`` when every record carries the run's own stamps."""
    records = [
        record
        for record in getattr(result, "evidence", ()) or ()
        if isinstance(record, EvidenceRecord)
    ]
    for record in records:
        for source_id in record.source_ids:
            if record.source_fingerprints.get(source_id) != observed.get(source_id):
                return []
    return records


def _with_provenance_events(
    package: KnowledgePackage,
    incompatible: Mapping[str, Sequence[str]],
    unattributed: Sequence[str],
) -> KnowledgePackage:
    """The package with one event per rejected source and per skipped run."""
    if not incompatible and not unattributed:
        return package
    events = list(package.events)
    event_ids = {event.event_id for event in events}
    status_by_source = {source.source_id: source.status for source in package.sources}

    def add(event: KnowledgeEvent) -> None:
        if event.event_id not in event_ids:
            events.append(event)
            event_ids.add(event.event_id)

    for source_id, runs in sorted(incompatible.items()):
        if source_id not in status_by_source:
            continue
        suffix = _domain_fingerprint(
            "fabric-rlm.knowledge.provenance-event.v1",
            {"source": source_id, "runs": sorted(runs)},
        )[:16]
        add(
            KnowledgeEvent(
                event_id=f"evidence.incompatible.{source_id}.{suffix}",
                event_type="evidence.incompatible",
                subject_type="source",
                subject_id=source_id,
                status=status_by_source[source_id],
                reason_code="schema_fingerprint_mismatch",
            )
        )
    for run in unattributed:
        suffix = _domain_fingerprint(
            "fabric-rlm.knowledge.provenance-event.v1",
            {"package": package.package_id, "run": run},
        )[:16]
        add(
            KnowledgeEvent(
                event_id=f"evidence.unattributed.{package.package_id}.{suffix}",
                event_type="evidence.unattributed",
                subject_type="package",
                subject_id=package.package_id,
                status="candidate",
                reason_code="no_recorded_source_fingerprints",
            )
        )
    return replace(package, events=tuple(events))


def _persist(
    store: KnowledgeStore,
    package: KnowledgePackage,
    *,
    transport: OneLakeKnowledgeTransport | None,
    overwrite: bool,
) -> None:
    location = _onelake_location(store)
    if location is None:
        if transport is not None:
            raise ValueError(
                "transport is only supported for OneLake knowledge stores"
            )
        save_knowledge_package(store, package, overwrite=overwrite)
    else:
        save_onelake_knowledge_package(
            location,
            package,
            transport=transport or OneLakeRestTransport(),
            overwrite=overwrite,
        )


def enrich_knowledge(
    knowledge: Knowledge,
    results: Sequence[object],
    *,
    store: KnowledgeStore | None = None,
    transport: OneLakeKnowledgeTransport | None = None,
    overwrite: bool = False,
    aliases: Mapping[str, str] | None = None,
) -> Knowledge:
    """A new ``Knowledge`` whose package has learned from finished runs.

    Evidence is harvested from each result's typed telemetry and outcome,
    attributed to the package's sources by the alias the run bound, and
    promoted into lessons by the per-kind policy. Nothing is written
    unless ``store`` is given, and the input ``knowledge`` is untouched;
    the caller decides when a learned package replaces a saved one.

    Evidence keeps the identity of the sources the run executed against
    (the schema fingerprints the runtime recorded on the trajectory, or
    the ones already on ``result.evidence``); it is never restamped with
    this package's. A record whose fingerprints do not match the package
    is dropped and noted as an ``evidence.incompatible`` event on its
    source; a result that carries no fingerprints at all (a run without a
    knowledge package bound) is skipped and noted as
    ``evidence.unattributed`` on the package.
    """
    if not isinstance(knowledge, Knowledge):
        raise TypeError("knowledge must be a Knowledge instance")
    package = knowledge.package
    known = [source.source_id for source in package.sources]
    alias_map = {str(key): str(value) for key, value in (aliases or {}).items()}
    harvested: list[EvidenceRecord] = []
    incompatible: dict[str, list[str]] = {}
    unattributed: list[str] = []
    for result in results:
        if not hasattr(getattr(result, "trajectory", None), "turns"):
            raise TypeError("results must contain RLMResult values")
        observed = _observed_fingerprints(result, alias_map)
        if not observed:
            unattributed.append(run_fingerprint_for(result))
            continue
        records = _reusable_evidence(result, observed) if not alias_map else []
        if not records:
            records = list(
                harvest_evidence(
                    result,
                    sources=knowledge.bindings,
                    known_source_ids=known,
                    source_fingerprints=observed,
                    aliases=aliases,
                )
            )
        current = {record.evidence_id for record in current_evidence(package, records)}
        for record in records:
            if record.evidence_id in current:
                harvested.append(record)
                continue
            for source_id in record.source_ids:
                runs = incompatible.setdefault(source_id, [])
                if record.run_fingerprint and record.run_fingerprint not in runs:
                    runs.append(record.run_fingerprint)
    package = _with_provenance_events(package, incompatible, unattributed)
    promoted = promote_lessons(package, harvested)
    if store is not None:
        _persist(store, promoted, transport=transport, overwrite=overwrite)
    return Knowledge(
        package=promoted,
        bindings=knowledge.bindings,
        _registry=knowledge._registry,
        _limits=knowledge._limits,
    )


def load_knowledge(
    source: KnowledgeStore,
    *,
    sources: Mapping[str, object],
    roles: Mapping[str, object] | None = None,
    limits: ProfileLimits | None = None,
    registry: SourceAdapterRegistry | None = None,
    transport: OneLakeKnowledgeTransport | None = None,
) -> Knowledge:
    """Load a portable package and explicitly bind freshly profiled sources."""

    location = _onelake_location(source)
    if location is None:
        if transport is not None:
            raise ValueError(
                "transport is only supported for OneLake knowledge stores"
            )
        package = read_knowledge_package(source)
    else:
        package = read_onelake_knowledge_package(
            location,
            transport=transport or OneLakeRestTransport(),
        )
    _exact_aliases(package, sources)
    active_roles = _load_roles(package, roles)
    active_limits = limits or ProfileLimits()
    active_registry = _active_registry(registry)
    profiles = profile_sources(
        sources,
        roles=active_roles,
        limits=active_limits,
        registry=active_registry,
    )
    _validate_current_profiles(package, profiles)
    bindings = _bindings_from_profiles(profiles, sources)
    bound = bind_knowledge_package(
        package,
        bindings=bindings,
    )
    return Knowledge(
        package=bound.package,
        bindings=bound.bindings,
        _registry=active_registry,
        _limits=active_limits,
    )


__all__ = ["Knowledge", "enrich_knowledge", "learn", "load_knowledge"]
