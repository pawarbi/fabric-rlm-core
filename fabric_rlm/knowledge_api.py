"""Public orchestration for learning and rebinding knowledge packages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os

from fabric_rlm.knowledge import (
    KnowledgePackage,
    SourceProfile,
    SourceRole,
    _domain_fingerprint,
)
from fabric_rlm.knowledge_lakehouse_sources import fabric_source_registry
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
) -> Knowledge:
    """Profile approved sources and return an immutable, runtime-bound package."""

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
    bindings = _bindings_from_profiles(profiles, sources)
    if store is not None:
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
    return Knowledge(
        package=package,
        bindings={alias: binding.value for alias, binding in bindings.items()},
        _registry=active_registry,
        _limits=active_limits,
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


__all__ = ["Knowledge", "learn", "load_knowledge"]
