"""Append-only evidence registry for experimental analysis runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fabric_rlm.experimental.analysis_contracts import EvidenceEntry
from fabric_rlm.experimental.analysis_reproducibility import fingerprint


_TERMINAL_STATES = {"completed", "failed", "rejected", "superseded"}
_TRANSITIONS = {
    "planned": {"running", "rejected", "superseded"},
    "running": {"completed", "failed", "rejected", "superseded"},
}


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _required_diagnostic_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(
        _required_text(value, f"required_diagnostics[{index}]")
        for index, value in enumerate(values)
    )
    if len(set(names)) != len(names):
        raise ValueError("required_diagnostics must not contain duplicates")
    return names


class EvidenceRegistry:
    """Record evidence lifecycle events without rewriting prior history."""

    def __init__(self, *, run_id: str) -> None:
        self._run_id = _required_text(run_id, "run_id")
        self._history: list[EvidenceEntry] = []
        self._current: dict[str, EvidenceEntry] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def history(self) -> tuple[EvidenceEntry, ...]:
        return tuple(self._history)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    def current(self, evidence_id: str) -> EvidenceEntry:
        identity = _required_text(evidence_id, "evidence_id")
        try:
            return self._current[identity]
        except KeyError as exc:
            raise KeyError(f"unknown evidence_id: {identity}") from exc

    def current_for_node(self, node_id: str) -> EvidenceEntry:
        identity = _required_text(node_id, "node_id")
        for entry in reversed(self._history):
            if entry.node_id == identity:
                return entry
        raise KeyError(f"no evidence found for node_id: {identity}")

    def append(self, entry: EvidenceEntry) -> None:
        if not isinstance(entry, EvidenceEntry):
            raise TypeError("entry must be an EvidenceEntry")

        prior = self._current.get(entry.evidence_id)
        if prior is None:
            if entry.state != "planned":
                raise ValueError(
                    f"evidence {entry.evidence_id} must start as planned"
                )
            self._validate_supersession(entry)
        else:
            if entry.node_id != prior.node_id:
                raise ValueError(
                    f"evidence {entry.evidence_id} node_id changed from "
                    f"{prior.node_id} to {entry.node_id}"
                )
            if prior.state in _TERMINAL_STATES:
                raise ValueError(
                    f"evidence {entry.evidence_id} is terminal in state "
                    f"{prior.state}"
                )
            allowed = _TRANSITIONS[prior.state]
            if entry.state not in allowed:
                raise ValueError(
                    f"evidence {entry.evidence_id} cannot transition from "
                    f"{prior.state} to {entry.state}"
                )
            if entry.supersedes != prior.supersedes:
                raise ValueError(
                    f"evidence {entry.evidence_id} supersedes must remain "
                    "unchanged during its lifecycle"
                )

        self._history.append(entry)
        self._current[entry.evidence_id] = entry

    def _validate_supersession(self, entry: EvidenceEntry) -> None:
        if entry.supersedes is None:
            return
        prior = self._current.get(entry.supersedes)
        if prior is None:
            raise ValueError(
                f"evidence {entry.evidence_id} supersedes missing evidence "
                f"{entry.supersedes}"
            )
        if prior.state not in _TERMINAL_STATES:
            raise ValueError(
                f"evidence {entry.evidence_id} cannot supersede nonterminal "
                f"evidence {entry.supersedes}"
            )

    def require_action_ready(
        self,
        evidence_ids: Sequence[str],
        *,
        required_diagnostics: Sequence[str] = (),
    ) -> tuple[EvidenceEntry, ...]:
        """Return completed evidence only when all required diagnostics pass."""

        identities = tuple(
            _required_text(value, f"evidence_ids[{index}]")
            for index, value in enumerate(evidence_ids)
        )
        if not identities:
            raise ValueError("evidence_ids must not be empty")
        if len(set(identities)) != len(identities):
            raise ValueError("evidence_ids must not contain duplicates")
        diagnostic_names = _required_diagnostic_names(required_diagnostics)

        ready: list[EvidenceEntry] = []
        for evidence_id in identities:
            entry = self._current.get(evidence_id)
            if entry is None:
                raise ValueError(f"unknown action evidence: {evidence_id}")
            if entry.state != "completed" or entry.result is None:
                raise ValueError(
                    f"action evidence {evidence_id} must be completed"
                )

            diagnostics = entry.result.diagnostics
            for name, diagnostic in diagnostics.items():
                if (
                    isinstance(diagnostic, Mapping)
                    and "passed" in diagnostic
                    and diagnostic["passed"] is not True
                ):
                    raise ValueError(
                        f"action evidence {evidence_id} "
                        f"diagnostics.{name} did not pass"
                    )
            for name in diagnostic_names:
                diagnostic = diagnostics.get(name)
                if (
                    not isinstance(diagnostic, Mapping)
                    or diagnostic.get("passed") is not True
                ):
                    raise ValueError(
                        f"action evidence {evidence_id} "
                        f"diagnostics.{name} did not pass"
                    )
            ready.append(entry)
        return tuple(ready)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "history": [entry.to_dict() for entry in self._history],
        }
