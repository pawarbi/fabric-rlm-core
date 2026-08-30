from __future__ import annotations

import pytest

from fabric_rlm.experimental import (
    EvidenceEntry,
    EvidenceRegistry,
    OperatorResult,
)


def _completed_result(
    *,
    node_id: str = "drivers",
    diagnostics_passed: bool = True,
) -> OperatorResult:
    return OperatorResult(
        node_id=node_id,
        operator="kpi.additive.v1",
        status="completed",
        seed=11,
        sample_size=100,
        values={"change": -0.05},
        diagnostics={
            "reconciliation": {
                "passed": diagnostics_passed,
                "residual": 0.0,
            }
        },
    )


def test_registry_preserves_append_only_lifecycle_history() -> None:
    registry = EvidenceRegistry(run_id="run-001")

    registry.append(
        EvidenceEntry(
            evidence_id="ev-drivers",
            node_id="drivers",
            state="planned",
        )
    )
    registry.append(
        EvidenceEntry(
            evidence_id="ev-drivers",
            node_id="drivers",
            state="running",
        )
    )
    registry.append(
        EvidenceEntry(
            evidence_id="ev-drivers",
            node_id="drivers",
            state="completed",
            result=_completed_result(),
        )
    )

    assert tuple(entry.state for entry in registry.history) == (
        "planned",
        "running",
        "completed",
    )
    assert registry.current("ev-drivers").state == "completed"
    assert registry.current_for_node("drivers").evidence_id == "ev-drivers"


@pytest.mark.parametrize(
    ("states", "match"),
    [
        (("running",), "must start as planned"),
        (("planned", "completed"), "planned.*completed"),
        (("planned", "running", "planned"), "running.*planned"),
        (("planned", "rejected", "running"), "terminal"),
    ],
)
def test_registry_rejects_invalid_lifecycle_transitions(
    states: tuple[str, ...],
    match: str,
) -> None:
    registry = EvidenceRegistry(run_id="run-001")

    with pytest.raises(ValueError, match=match):
        for state in states:
            registry.append(
                EvidenceEntry(
                    evidence_id="ev-drivers",
                    node_id="drivers",
                    state=state,
                    result=_completed_result() if state == "completed" else None,
                )
            )


def test_registry_rejects_node_identity_changes_within_lifecycle() -> None:
    registry = EvidenceRegistry(run_id="run-001")
    registry.append(
        EvidenceEntry(
            evidence_id="ev-drivers",
            node_id="drivers",
            state="planned",
        )
    )

    with pytest.raises(ValueError, match="node_id"):
        registry.append(
            EvidenceEntry(
                evidence_id="ev-drivers",
                node_id="different-node",
                state="running",
            )
        )


def test_registry_requires_completed_passing_diagnostics_for_action_evidence() -> None:
    registry = EvidenceRegistry(run_id="run-001")
    for evidence_id, passed in (("ev-passing", True), ("ev-failing", False)):
        registry.append(
            EvidenceEntry(
                evidence_id=evidence_id,
                node_id=evidence_id.removeprefix("ev-"),
                state="planned",
            )
        )
        registry.append(
            EvidenceEntry(
                evidence_id=evidence_id,
                node_id=evidence_id.removeprefix("ev-"),
                state="running",
            )
        )
        registry.append(
            EvidenceEntry(
                evidence_id=evidence_id,
                node_id=evidence_id.removeprefix("ev-"),
                state="completed",
                result=_completed_result(
                    node_id=evidence_id.removeprefix("ev-"),
                    diagnostics_passed=passed,
                ),
            )
        )

    ready = registry.require_action_ready(
        ("ev-passing",),
        required_diagnostics=("reconciliation",),
    )
    assert tuple(entry.evidence_id for entry in ready) == ("ev-passing",)

    with pytest.raises(ValueError, match="ev-failing.*diagnostics.reconciliation"):
        registry.require_action_ready(
            ("ev-failing",),
            required_diagnostics=("reconciliation",),
        )


def test_registry_rejects_unknown_or_noncompleted_action_evidence() -> None:
    registry = EvidenceRegistry(run_id="run-001")
    registry.append(
        EvidenceEntry(
            evidence_id="ev-pending",
            node_id="pending",
            state="planned",
        )
    )

    with pytest.raises(ValueError, match="ev-missing"):
        registry.require_action_ready(("ev-missing",))
    with pytest.raises(ValueError, match="ev-pending.*completed"):
        registry.require_action_ready(("ev-pending",))
    with pytest.raises(ValueError, match="must not be empty"):
        registry.require_action_ready(())
    with pytest.raises(ValueError, match="duplicates"):
        registry.require_action_ready(("ev-pending", "ev-pending"))

    registry.append(
        EvidenceEntry(
            evidence_id="ev-failed",
            node_id="failed",
            state="planned",
        )
    )
    registry.append(
        EvidenceEntry(
            evidence_id="ev-failed",
            node_id="failed",
            state="running",
        )
    )
    registry.append(
        EvidenceEntry(
            evidence_id="ev-failed",
            node_id="failed",
            state="failed",
            result=OperatorResult(
                node_id="failed",
                operator="kpi.additive.v1",
                status="failed",
                seed=1,
                sample_size=0,
                failure_code="insufficient_data",
                failure_message="No valid comparison rows",
            ),
        )
    )
    with pytest.raises(ValueError, match="ev-failed.*completed"):
        registry.require_action_ready(("ev-failed",))


def test_registry_validates_supersession_without_mutating_prior_evidence() -> None:
    registry = EvidenceRegistry(run_id="run-001")
    registry.append(
        EvidenceEntry(
            evidence_id="ev-original",
            node_id="drivers",
            state="planned",
        )
    )
    registry.append(
        EvidenceEntry(
            evidence_id="ev-original",
            node_id="drivers",
            state="rejected",
        )
    )
    registry.append(
        EvidenceEntry(
            evidence_id="ev-replacement",
            node_id="drivers-v2",
            state="planned",
            supersedes="ev-original",
        )
    )

    assert registry.current("ev-original").state == "rejected"
    assert registry.current("ev-replacement").supersedes == "ev-original"

    with pytest.raises(ValueError, match="supersedes.*missing"):
        registry.append(
            EvidenceEntry(
                evidence_id="ev-invalid",
                node_id="invalid",
                state="planned",
                supersedes="ev-missing",
            )
        )

    live_registry = EvidenceRegistry(run_id="run-002")
    live_registry.append(
        EvidenceEntry(
            evidence_id="ev-live",
            node_id="live",
            state="planned",
        )
    )
    with pytest.raises(ValueError, match="nonterminal"):
        live_registry.append(
            EvidenceEntry(
                evidence_id="ev-new",
                node_id="new",
                state="planned",
                supersedes="ev-live",
            )
        )


def test_registry_rejects_supersedes_mutation_during_lifecycle() -> None:
    registry = EvidenceRegistry(run_id="run-001")
    for evidence_id in ("ev-a", "ev-b"):
        registry.append(
            EvidenceEntry(
                evidence_id=evidence_id,
                node_id=evidence_id,
                state="planned",
            )
        )
        registry.append(
            EvidenceEntry(
                evidence_id=evidence_id,
                node_id=evidence_id,
                state="rejected",
            )
        )

    registry.append(
        EvidenceEntry(
            evidence_id="ev-new",
            node_id="new",
            state="planned",
            supersedes="ev-a",
        )
    )
    with pytest.raises(ValueError, match="supersedes.*unchanged"):
        registry.append(
            EvidenceEntry(
                evidence_id="ev-new",
                node_id="new",
                state="running",
                supersedes="ev-b",
            )
        )


def test_registry_export_and_fingerprint_are_deterministic() -> None:
    def build_registry() -> EvidenceRegistry:
        registry = EvidenceRegistry(run_id="run-001")
        registry.append(
            EvidenceEntry(
                evidence_id="ev-drivers",
                node_id="drivers",
                state="planned",
            )
        )
        registry.append(
            EvidenceEntry(
                evidence_id="ev-drivers",
                node_id="drivers",
                state="running",
            )
        )
        registry.append(
            EvidenceEntry(
                evidence_id="ev-drivers",
                node_id="drivers",
                state="completed",
                result=_completed_result(),
            )
        )
        return registry

    first = build_registry()
    second = build_registry()

    assert first.to_dict() == second.to_dict()
    assert first.fingerprint == second.fingerprint
