"""Passive evidence capture from typed runtime telemetry.

A finished run already knows a great deal about how its sources behaved:
which grain each query asked for, how many groups the preflight estimated,
whether the query ran, was rejected or timed out, how long it took, and
whether the answer that rested on it passed verification and the
analytical-integrity screen. This module turns those typed records into
:class:`~fabric_rlm.knowledge.EvidenceRecord` values.

It reads only what the runtime recorded as data: ``TurnRecord.source_calls``
(worker-side semantic-model telemetry and parent-side Lakehouse timings),
the run outcome, and the trajectory's typed error classes. It never mines
stdout or agent prose, and it never copies a data value: filter values,
result rows and DAX text stay out; names, counts, codes and timings go in.

Capture is observational. Nothing here changes a prompt or an execution;
the same run with capture on or off produces the same answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import TYPE_CHECKING, Any

from fabric_rlm.knowledge import EvidenceRecord, _domain_fingerprint
from fabric_rlm.trajectory import TurnRecord

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from fabric_rlm.runtime import RLMResult

# Telemetry keys that carry data, paths or free text and must not persist.
_DROPPED_TELEMETRY_KEYS = frozenset({"query", "error", "input", "source_root"})
_UNKNOWN_MEASURE = re.compile(r"Unknown semantic-model measure: (?P<name>[^\n]+)")
_UNKNOWN_COLUMN = re.compile(
    r"Unknown semantic-model column \((?P<role>[a-z]+)\): (?P<name>[^\n]+)"
)
_ERROR_CLASS = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_.]*)(?::|$)")
_MAX_OBSERVATION_LIST = 50


def _identifier_safe(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)) and ".." not in name


def _error_class(error: str | None) -> str | None:
    """The exception class of a recorded error, without its message.

    Telemetry records the class first (``ValueError: ...``); a traceback
    puts it on its last line. Both shapes are read, neither message is.
    """
    if not error:
        return None
    lines = [line.strip() for line in error.strip().splitlines() if line.strip()]
    for candidate in (lines[0], lines[-1]):
        match = _ERROR_CLASS.match(candidate)
        if match and ":" in candidate:
            return match.group("name").rsplit(".", 1)[-1]
    match = _ERROR_CLASS.match(lines[-1])
    return match.group("name").rsplit(".", 1)[-1] if match else None


def _execution_status(record: Mapping[str, Any]) -> str:
    reason = record.get("reason")
    if reason in {"cardinality_limit", "validation"}:
        return "rejected"
    if reason == "preflight_timeout":
        return "timeout"
    if reason in {"execution_error", "preflight_error"}:
        error_class = _error_class(record.get("error")) or ""
        return "timeout" if "Timeout" in error_class else "failure"
    if record.get("executed") is False:
        return "rejected"
    return "success"


def _clean_value(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return None
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip().splitlines()[0] if value.strip() else ""
        return text[:256]
    if isinstance(value, Mapping):
        return {
            str(key): _clean_value(item, depth + 1)
            for key, item in list(value.items())[:_MAX_OBSERVATION_LIST]
            if isinstance(key, str)
        }
    if isinstance(value, (list, tuple)):
        return [_clean_value(item, depth + 1) for item in list(value)[:_MAX_OBSERVATION_LIST]]
    return str(value)[:256]


def _query_observation(record: Mapping[str, Any]) -> dict[str, Any]:
    """The persistable part of one telemetry record."""
    observation: dict[str, Any] = {}
    for key, value in record.items():
        if key in _DROPPED_TELEMETRY_KEYS or not isinstance(key, str):
            continue
        cleaned = _clean_value(value)
        if cleaned is not None:
            observation[key] = cleaned
    grain = record.get("groupby")
    if isinstance(grain, (list, tuple)):
        observation["grain"] = sorted(str(item) for item in grain)
    error_class = _error_class(record.get("error"))
    if error_class:
        observation["error_class"] = error_class
    error_text = record.get("error") or ""
    unknown_measure = _UNKNOWN_MEASURE.search(error_text)
    unknown_column = _UNKNOWN_COLUMN.search(error_text)
    if unknown_measure:
        observation["invalid_reference"] = unknown_measure.group("name").strip()[:256]
        observation["invalid_reference_kind"] = "measure"
    elif unknown_column:
        observation["invalid_reference"] = unknown_column.group("name").strip()[:256]
        observation["invalid_reference_kind"] = "column"
    return observation


def _source_roots(sources: Mapping[str, object] | None) -> dict[str, str]:
    roots: dict[str, str] = {}
    for alias, value in (sources or {}).items():
        root = getattr(value, "root", None)
        if isinstance(root, str) and root and isinstance(alias, str):
            roots.setdefault(root, alias)
    return roots


def _resolve_source_id(
    record: Mapping[str, Any],
    *,
    known_ids: set[str] | None,
    aliases: Mapping[str, str],
    roots: Mapping[str, str],
) -> str | None:
    """Map a telemetry record to the package source it belongs to.

    Worker records name the namespace input (``arr_model`` or
    ``sources.arr`` for a nested one); Lakehouse records name the root.
    The alias a task bound is the package's source id, so the top-level
    input name is the id; ``aliases`` can override that mapping for a run
    that bound a source under a different name.
    """
    name = record.get("input")
    if not isinstance(name, str) or not name:
        root = record.get("source_root")
        name = roots.get(str(root), "") if root else ""
    if not name:
        return None
    top_level = re.split(r"[.\[]", name, maxsplit=1)[0]
    candidates = [aliases.get(name), aliases.get(top_level), name, top_level]
    for candidate in candidates:
        if not candidate or not _identifier_safe(candidate):
            continue
        if known_ids is None or candidate in known_ids:
            return candidate
    return None


def run_fingerprint_for(result: RLMResult) -> str:
    """A stable identifier for one run, derived from what it executed."""
    turns = [
        {"turn": turn.turn, "code": turn.code, "submitted": turn.submitted}
        for turn in result.trajectory.turns
    ]
    metadata = result.trajectory.metadata or {}
    return "run." + _domain_fingerprint(
        "fabric-rlm.knowledge.run.v1",
        {
            "turns": turns,
            "knowledge_fingerprint": metadata.get("knowledge_fingerprint"),
            "payload": _clean_value(result.payload),
        },
    )[:20]


def verifier_status_for(result: RLMResult) -> str:
    """What verification said about the run's final answer."""
    if result.submitted and result.failure_reason is None:
        return "passed"
    return "failed"


def integrity_status_for(result: RLMResult) -> str:
    metadata = result.trajectory.metadata or {}
    mode = metadata.get("analytical_integrity_mode")
    if mode in {False, None, "off"} and "analytical_integrity_mode" in metadata:
        return "off"
    if metadata.get("analytical_integrity_unresolved"):
        return "unresolved"
    if not result.submitted:
        return "failed"
    return "passed"


def _evidence_id(payload: Mapping[str, Any]) -> str:
    return "evidence." + _domain_fingerprint("fabric-rlm.knowledge.evidence.v1", payload)[:20]


def harvest_evidence(
    result: RLMResult,
    *,
    sources: Mapping[str, object] | None = None,
    known_source_ids: Sequence[str] | None = None,
    source_fingerprints: Mapping[str, str] | None = None,
    aliases: Mapping[str, str] | None = None,
    run_fingerprint: str | None = None,
) -> tuple[EvidenceRecord, ...]:
    """Evidence records for one finished run.

    ``sources`` are the inputs the run bound (alias to handle), used to
    attribute Lakehouse records by root and to know which aliases carried
    evidence; ``known_source_ids`` restricts attribution to a package's
    sources so a record for an unbound alias is dropped rather than
    invented; ``source_fingerprints`` (source id to schema fingerprint) is
    stamped on every record so a later schema change can stale what rests
    on it; ``aliases`` maps a run's input names to package source ids when
    they differ.
    """
    trajectory = getattr(result, "trajectory", None)
    if trajectory is None or not hasattr(trajectory, "turns"):
        raise TypeError("harvest_evidence expects an RLMResult")
    known = set(known_source_ids) if known_source_ids is not None else None
    alias_map = {str(k): str(v) for k, v in (aliases or {}).items()}
    roots = _source_roots(sources)
    fingerprint = run_fingerprint or run_fingerprint_for(result)
    verifier = verifier_status_for(result)
    integrity = integrity_status_for(result)
    fingerprints = {
        str(k): str(v) for k, v in (source_fingerprints or {}).items()
    }

    def stamped(source_ids: Sequence[str]) -> dict[str, str]:
        return {
            source_id: fingerprints[source_id]
            for source_id in source_ids
            if source_id in fingerprints
        }

    records: list[EvidenceRecord] = []
    seen_ids: set[str] = set()

    def add(record: EvidenceRecord) -> None:
        if record.evidence_id not in seen_ids:
            seen_ids.add(record.evidence_id)
            records.append(record)

    successful_grains: list[tuple[str, list[str], bool]] = []
    touched_sources: list[str] = []
    for turn in result.trajectory.turns:
        for raw in turn.source_calls or ():
            if not isinstance(raw, Mapping):
                continue
            source_id = _resolve_source_id(
                raw, known_ids=known, aliases=alias_map, roots=roots
            )
            if source_id is None:
                continue
            if source_id not in touched_sources:
                touched_sources.append(source_id)
            observation = _query_observation(raw)
            status = _execution_status(raw)
            payload = {
                "source_ids": [source_id],
                "observation_type": "query_execution",
                "observation": observation,
                "turn": turn.turn,
                "run": fingerprint,
                "status": status,
            }
            add(
                EvidenceRecord(
                    evidence_id=_evidence_id(payload),
                    evidence_type="execution",
                    source_ids=(source_id,),
                    observation_type="query_execution",
                    observation=observation,
                    source_fingerprints=stamped([source_id]),
                    execution_status=status,
                    verifier_status=verifier,
                    analytical_integrity_status=integrity,
                    run_fingerprint=fingerprint,
                    turn=turn.turn,
                )
            )
            if status == "success" and observation.get("query_type") in {"aggregate", "measure"}:
                successful_grains.append(
                    (
                        source_id,
                        list(observation.get("grain") or []),
                        bool(observation.get("filter_count")),
                    )
                )

    for source_id in touched_sources:
        steps = [
            {"grain": grain, "filtered": filtered}
            for candidate, grain, filtered in successful_grains
            if candidate == source_id
        ]
        if len(steps) < 2:
            continue
        observation = {"steps": steps[:_MAX_OBSERVATION_LIST], "step_count": len(steps)}
        payload = {
            "source_ids": [source_id],
            "observation_type": "strategy_sequence",
            "observation": observation,
            "run": fingerprint,
        }
        add(
            EvidenceRecord(
                evidence_id=_evidence_id(payload),
                evidence_type="trajectory",
                source_ids=(source_id,),
                observation_type="strategy_sequence",
                observation=observation,
                source_fingerprints=stamped([source_id]),
                execution_status="success",
                verifier_status=verifier,
                analytical_integrity_status=integrity,
                run_fingerprint=fingerprint,
            )
        )

    bound_ids = [
        alias
        for alias in (sources or {})
        if isinstance(alias, str)
        and _identifier_safe(alias)
        and (known is None or alias in known)
    ]
    outcome_sources = touched_sources or bound_ids
    if outcome_sources:
        turns = result.trajectory.turns
        source_call_count = sum(len(turn.source_calls or ()) for turn in turns)
        failed_calls = sum(
            1
            for turn in turns
            for raw in (turn.source_calls or ())
            if isinstance(raw, Mapping) and _execution_status(raw) != "success"
        )
        first_useful = next(
            (
                turn.turn
                for turn in turns
                if any(
                    isinstance(raw, Mapping)
                    and _execution_status(raw) == "success"
                    and (raw.get("returned_rows") or 0) > 0
                    for raw in (turn.source_calls or ())
                )
            ),
            None,
        )
        metadata = result.trajectory.metadata or {}
        observation = {
            "turns": len(turns),
            "submitted": bool(result.submitted),
            "failure_reason": str(result.failure_reason or "none")[:64],
            "verifier_repairs": len(metadata.get("verifier_repair_history") or []),
            "integrity_findings": len(metadata.get("analytical_integrity_unresolved") or []),
            "source_calls": source_call_count,
            "failed_source_calls": failed_calls,
            "error_turns": sum(1 for turn in turns if turn.error),
            "first_useful_query_turn": first_useful,
            "error_classes": sorted(
                {
                    cls
                    for cls in (_error_class(turn.error) for turn in turns)
                    if cls
                }
            )[:_MAX_OBSERVATION_LIST],
        }
        payload = {
            "source_ids": sorted(outcome_sources),
            "observation_type": "run_outcome",
            "observation": observation,
            "run": fingerprint,
        }
        add(
            EvidenceRecord(
                evidence_id=_evidence_id(payload),
                evidence_type="trajectory",
                source_ids=tuple(sorted(outcome_sources)),
                observation_type="run_outcome",
                observation=observation,
                source_fingerprints=stamped(sorted(outcome_sources)),
                execution_status="success" if result.submitted else "failure",
                verifier_status=verifier,
                analytical_integrity_status=integrity,
                run_fingerprint=fingerprint,
            )
        )
    return tuple(records)


def source_call_summary(turns: Sequence[TurnRecord]) -> dict[str, Any]:
    """Counts a benchmark reads off a trajectory without harvesting."""
    calls = [
        raw
        for turn in turns
        for raw in (turn.source_calls or ())
        if isinstance(raw, Mapping)
    ]
    seconds = 0.0
    for raw in calls:
        for key in ("total_seconds", "execution_seconds", "preflight_seconds"):
            value = raw.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                seconds += float(value)
                break
    first_useful = next(
        (
            turn.turn
            for turn in turns
            if any(
                isinstance(raw, Mapping)
                and _execution_status(raw) == "success"
                and (raw.get("returned_rows") or 0) > 0
                for raw in (turn.source_calls or ())
            )
        ),
        None,
    )
    return {
        "source_calls": len(calls),
        "failed_source_calls": sum(1 for raw in calls if _execution_status(raw) != "success"),
        "source_seconds": round(seconds, 3),
        "first_useful_query_turn": first_useful,
    }


__all__ = [
    "harvest_evidence",
    "integrity_status_for",
    "run_fingerprint_for",
    "source_call_summary",
    "verifier_status_for",
]
