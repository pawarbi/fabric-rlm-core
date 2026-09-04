"""Lesson derivation and promotion: evidence in, structured lessons out.

Two kinds of lesson exist. Structural lessons come straight from what a
source declares about itself (a semantic model with an explicit
current-period construct) and are active from the start. Evidence lessons
are promoted from :class:`~fabric_rlm.knowledge.EvidenceRecord` values by a
per-kind policy: a query rejected by the cardinality preflight proves an
expensive grain at once, one timeout only nominates it, a strategy is
preferred only after verified runs used it, and a measure that equals its
base under an unfiltered context nominates a context requirement that a
contrasting filtered observation confirms. Causal or business
interpretations are never derived here.

Lessons carry the evidence they rest on and the schema fingerprints they
depend on; evidence captured against a different schema is kept but does
not promote. Quarantined and retired lessons stay that way.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
import re
from typing import Any

from fabric_rlm.knowledge import (
    EvidenceRecord,
    KnowledgeEvent,
    KnowledgePackage,
    LearnedLesson,
    SourceProfile,
    _domain_fingerprint,
)


_CURRENT_PERIOD = re.compile(
    r"(?i)(?:^|[^a-z])(?:is[_ ]?)?current(?:[_ ]?(?:year|quarter|month|period|week|date|fy|fq|yq|ym))?"
    r"(?:$|[^a-z])|(?:^|[^a-z])as[_ ]?of(?:$|[^a-z])|(?:^|[^a-z])latest[_ ]?(?:period|quarter|month|date|week)"
)
_DERIVED_TIME_MEASURE = re.compile(
    r"(?i)previous|prior|(?:^|[^a-z])py(?:$|[^a-z])|(?:^|[^a-z])pp(?:$|[^a-z])|yoy|qoq|mom|growth|"
    r"(?:^|[^a-z])nrr(?:$|[^a-z])|(?:^|[^a-z])grr(?:$|[^a-z])|retention|churn|(?:^|[^a-z])change|"
    r"delta|variance|(?:^|[^a-z])vs(?:$|[^a-z])|last (?:year|quarter|month|period)|ttm|ltm|ytd|qtd|mtd"
)
_MAX_CONSTRUCTS = 10
_MAX_MEASURES_PER_GRAIN = 10
_MAX_CONTEXT_CANDIDATES = 40


def _leaf(name: str) -> str:
    """``Period[IsCurrentQuarter]`` gives ``IsCurrentQuarter``."""
    text = str(name).strip()
    if text.endswith("]") and "[" in text:
        return text[text.rindex("[") + 1:-1].strip() or text
    return text


def _lesson_id(kind: str, source_id: str, subject: str) -> str:
    suffix = _domain_fingerprint(
        "fabric-rlm.knowledge.lesson.v1",
        {"kind": kind, "source_id": source_id, "subject": subject.strip().lower()},
    )[:16]
    return f"lesson.{kind}.{suffix}"


def _grain_subject(grain: Sequence[str]) -> str:
    return " x ".join(_leaf(item) for item in grain) or "total"


def _schema_names(profile: SourceProfile, family: str) -> list[str]:
    section = profile.schema.get(family) if isinstance(profile.schema, Mapping) else None
    if not isinstance(section, Mapping):
        return []
    return [str(name) for name in section]


def structural_lessons(package: KnowledgePackage) -> tuple[LearnedLesson, ...]:
    """Lessons a source declares about itself, active on arrival.

    A semantic model with a current-period construct (``Period[IsCurrentQuarter]``,
    ``Calendar[AsOfDate]``) defines "current" itself; an agent must not
    infer it from the maximum date. Measures whose names mark them as
    time-relative or derived (previous period, growth, retention) are
    nominated, as candidates only, for a context requirement that evidence
    can later confirm.
    """
    lessons: list[LearnedLesson] = []
    for profile in package.sources:
        if profile.family != "semantic_model":
            continue
        fingerprints = {profile.source_id: profile.schema_fingerprint}
        columns = _schema_names(profile, "columns")
        measures = _schema_names(profile, "measures")
        constructs = [
            name
            for name in columns + measures
            if _CURRENT_PERIOD.search(_leaf(name))
        ]
        if constructs:
            lessons.append(
                LearnedLesson(
                    lesson_id=_lesson_id("time_semantics", profile.source_id, "current period"),
                    kind="time_semantics",
                    subject="current period",
                    structured_rule={
                        "current_period_constructs": sorted(constructs)[:_MAX_CONSTRUCTS],
                        "rule": "use_declared_current_period",
                        "avoid": "max_date_inference",
                    },
                    confidence="high",
                    status="active",
                    source_dependencies=(profile.source_id,),
                    source_fingerprints=fingerprints,
                    basis=("source_declared",),
                )
            )
        derived = [name for name in measures if _DERIVED_TIME_MEASURE.search(_leaf(name))]
        for name in sorted(derived)[:_MAX_CONTEXT_CANDIDATES]:
            measure = _leaf(name)
            lessons.append(
                LearnedLesson(
                    lesson_id=_lesson_id("context_requirement", profile.source_id, measure),
                    kind="context_requirement",
                    subject=measure,
                    structured_rule={
                        "measure": measure,
                        "requires": ["period_context"],
                        "fallback_strategy": "explicit_period_base_measure",
                        "observed": "name_pattern",
                    },
                    confidence="low",
                    status="candidate",
                    source_dependencies=(profile.source_id,),
                    source_fingerprints=fingerprints,
                    basis=("name_pattern",),
                )
            )
    return tuple(lessons)


def _current_evidence(
    package: KnowledgePackage,
    evidence: Iterable[EvidenceRecord],
) -> list[EvidenceRecord]:
    """Evidence whose stamped schema fingerprints match the package's sources.

    A record captured against an older schema is not thrown away, but it
    does not promote anything against the schema that replaced it.
    """
    current = {source.source_id: source.schema_fingerprint for source in package.sources}
    kept: list[EvidenceRecord] = []
    for record in evidence:
        if any(source_id not in current for source_id in record.source_ids):
            continue
        stale = any(
            record.source_fingerprints.get(source_id) not in {None, current[source_id]}
            for source_id in record.source_ids
        )
        if not stale:
            kept.append(record)
    return kept


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _expensive_grain_lessons(
    source_id: str,
    fingerprints: Mapping[str, str],
    records: Sequence[EvidenceRecord],
) -> dict[str, LearnedLesson]:
    by_grain: dict[tuple[str, ...], list[EvidenceRecord]] = {}
    for record in records:
        if record.observation_type != "query_execution":
            continue
        grain = record.observation.get("grain")
        if not grain or record.execution_status not in {"timeout", "rejected"}:
            continue
        if record.execution_status == "rejected" and record.observation.get("reason") != "cardinality_limit":
            continue
        by_grain.setdefault(tuple(sorted(str(item) for item in grain)), []).append(record)
    lessons: dict[str, LearnedLesson] = {}
    for grain, group in by_grain.items():
        cardinality = [r for r in group if r.observation.get("reason") == "cardinality_limit"]
        timeouts = [r for r in group if r.execution_status == "timeout"]
        estimates = [
            _number(r.observation.get("estimated_groups")) for r in cardinality
        ]
        estimates = [e for e in estimates if e is not None]
        limits = [_number(r.observation.get("max_groups")) for r in group]
        limits = [l for l in limits if l is not None]
        if cardinality:
            status, confidence, basis = "active", "high", ("preflight_estimate",)
            outcome = "cardinality_limit"
        elif len(timeouts) >= 2:
            status, confidence, basis = "active", "medium", ("repeated_timeout",)
            outcome = "preflight_timeout"
        else:
            status, confidence, basis = "candidate", "low", ("single_timeout",)
            outcome = "preflight_timeout"
        rule: dict[str, Any] = {
            "grain": list(grain),
            "grain_size": len(grain),
            "outcome": outcome,
            "observations": len(group),
            "advice": "narrow_grain_or_filter_first",
        }
        if estimates:
            rule["estimated_groups"] = int(max(estimates))
        if limits:
            rule["max_groups"] = int(max(limits))
        measure_counts = [_number(r.observation.get("measure_count")) for r in group]
        measure_counts = [m for m in measure_counts if m is not None]
        if measure_counts:
            rule["measure_count"] = int(max(measure_counts))
        subject = _grain_subject(grain)
        lesson_id = _lesson_id("expensive_grain", source_id, subject)
        lessons[lesson_id] = LearnedLesson(
            lesson_id=lesson_id,
            kind="expensive_grain",
            subject=subject,
            structured_rule=rule,
            evidence_ids=tuple(sorted(r.evidence_id for r in group)),
            confidence=confidence,
            status=status,
            source_dependencies=(source_id,),
            source_fingerprints=fingerprints,
            basis=basis,
        )
    return lessons


def _valid_grain_lessons(
    source_id: str,
    fingerprints: Mapping[str, str],
    records: Sequence[EvidenceRecord],
    expensive_grains: set[tuple[str, ...]],
) -> dict[str, LearnedLesson]:
    by_grain: dict[tuple[str, ...], list[EvidenceRecord]] = {}
    for record in records:
        if record.observation_type != "query_execution" or record.execution_status != "success":
            continue
        if record.observation.get("query_type") not in {"aggregate", "measure"}:
            continue
        grain = record.observation.get("grain")
        rows = _number(record.observation.get("returned_rows"))
        if not grain or rows is None or rows <= 0:
            continue
        by_grain.setdefault(tuple(sorted(str(item) for item in grain)), []).append(record)
    lessons: dict[str, LearnedLesson] = {}
    for grain, group in by_grain.items():
        trusted = [r for r in group if r.trusted]
        if len(group) >= 2 and trusted:
            status, confidence = "active", ("high" if len(trusted) >= 2 else "medium")
        else:
            status, confidence = "candidate", "low"
        basis = ["verified_success"] if trusted else ["single_success"]
        reason_code = None
        if grain in expensive_grains:
            # The same grain also timed out or was rejected: the failure
            # evidence wins for safety, and the claim waits for review.
            status, reason_code = "quarantined", "contradicting_evidence"
            basis.append("contradicted_by_failure")
        rows = [_number(r.observation.get("returned_rows")) for r in group]
        seconds = [
            _number(r.observation.get("total_seconds"))
            or _number(r.observation.get("execution_seconds"))
            for r in group
        ]
        measures: list[str] = []
        for r in group:
            for name in r.observation.get("measures") or ():
                if isinstance(name, str) and name not in measures:
                    measures.append(name)
        rule: dict[str, Any] = {
            "grain": list(grain),
            "grain_size": len(grain),
            "measures": measures[:_MAX_MEASURES_PER_GRAIN],
            "successes": len(group),
            "verified_successes": len(trusted),
            "advice": "reliable_analysis_grain",
        }
        if any(r is not None for r in rows):
            rule["max_rows_observed"] = int(max(r for r in rows if r is not None))
        if any(s is not None for s in seconds):
            rule["max_seconds_observed"] = round(max(s for s in seconds if s is not None), 3)
        subject = _grain_subject(grain)
        lesson_id = _lesson_id("valid_grain", source_id, subject)
        lessons[lesson_id] = LearnedLesson(
            lesson_id=lesson_id,
            kind="valid_grain",
            subject=subject,
            structured_rule=rule,
            evidence_ids=tuple(sorted(r.evidence_id for r in group)),
            confidence=confidence,
            status=status,
            source_dependencies=(source_id,),
            source_fingerprints=fingerprints,
            basis=tuple(basis),
            reason_code=reason_code,
        )
    return lessons


def _is_derived_measure(name: str) -> bool:
    return bool(_DERIVED_TIME_MEASURE.search(_leaf(name)))


def _context_requirement_lessons(
    source_id: str,
    fingerprints: Mapping[str, str],
    records: Sequence[EvidenceRecord],
) -> dict[str, LearnedLesson]:
    """A derived measure that collapses without a period context.

    Under an unfiltered context ``ARR $ Previous Period`` returned exactly
    ``ARR $`` (or a growth rate returned a constant zero); with a period
    filter the two differ. The first observation nominates, the contrast
    confirms, and repeated degenerate observations across runs confirm
    too.
    """
    degenerate: dict[str, list[tuple[EvidenceRecord, str | None, str | None]]] = {}
    distinct: dict[str, list[EvidenceRecord]] = {}
    for record in records:
        if record.observation_type != "query_execution" or record.execution_status != "success":
            continue
        observation = record.observation
        measures = [m for m in (observation.get("measures") or ()) if isinstance(m, str)]
        unfiltered = not observation.get("filter_count")
        identities = observation.get("measure_identities") or ()
        identical: dict[str, str] = {}
        for pair in identities:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            left, right = str(pair[0]), str(pair[1])
            if _is_derived_measure(left) and not _is_derived_measure(right):
                identical[left] = right
            elif _is_derived_measure(right) and not _is_derived_measure(left):
                identical[right] = left
        constants = observation.get("constant_measures") or {}
        if unfiltered:
            for measure, base in identical.items():
                degenerate.setdefault(measure, []).append((record, base, None))
            for measure, constant in constants.items():
                if _is_derived_measure(str(measure)) and constant in {"zero", "one"}:
                    degenerate.setdefault(str(measure), []).append((record, None, str(constant)))
        else:
            for measure in measures:
                if _is_derived_measure(measure) and measure not in identical and (
                    constants.get(measure) not in {"zero", "one"}
                ):
                    distinct.setdefault(measure, []).append(record)
    lessons: dict[str, LearnedLesson] = {}
    for measure, observations in degenerate.items():
        contrasts = distinct.get(measure, [])
        runs = {r.run_fingerprint for r, _b, _c in observations}
        if contrasts:
            status, confidence, basis = "active", "high", ("degenerate_unfiltered", "distinct_when_filtered")
        elif len(runs) >= 2:
            status, confidence, basis = "active", "medium", ("degenerate_unfiltered", "repeated_runs")
        else:
            status, confidence, basis = "candidate", "medium", ("degenerate_unfiltered",)
        bases = [b for _r, b, _c in observations if b]
        constant_codes = [c for _r, _b, c in observations if c]
        rule: dict[str, Any] = {
            "measure": measure,
            "requires": ["period_context"],
            "observed": "identity_under_unfiltered_context" if bases else "constant_under_unfiltered_context",
            "fallback_strategy": "explicit_period_base_measure",
            "observations": len(observations),
            "contrasting_observations": len(contrasts),
        }
        if bases:
            rule["base_measure"] = sorted(set(bases))[0]
        if constant_codes:
            rule["constant"] = sorted(set(constant_codes))[0]
        lesson_id = _lesson_id("context_requirement", source_id, measure)
        evidence_ids = sorted(
            {r.evidence_id for r, _b, _c in observations} | {r.evidence_id for r in contrasts}
        )
        lessons[lesson_id] = LearnedLesson(
            lesson_id=lesson_id,
            kind="context_requirement",
            subject=measure,
            structured_rule=rule,
            evidence_ids=tuple(evidence_ids),
            confidence=confidence,
            status=status,
            source_dependencies=(source_id,),
            source_fingerprints=fingerprints,
            basis=basis,
        )
    return lessons


def _metric_equivalence_lessons(
    source_id: str,
    fingerprints: Mapping[str, str],
    records: Sequence[EvidenceRecord],
) -> dict[str, LearnedLesson]:
    """Two measures identical across several filtered contexts, never apart.

    An identity under an unfiltered context does not count: that is the
    signature of a missing context, not of equivalence.
    """
    identical: dict[tuple[str, str], list[EvidenceRecord]] = {}
    apart: set[tuple[str, str]] = set()
    for record in records:
        if record.observation_type != "query_execution" or record.execution_status != "success":
            continue
        observation = record.observation
        if not observation.get("filter_count"):
            continue
        measures = [m for m in (observation.get("measures") or ()) if isinstance(m, str)]
        pairs = {
            tuple(sorted((str(p[0]), str(p[1]))))
            for p in (observation.get("measure_identities") or ())
            if isinstance(p, (list, tuple)) and len(p) == 2
        }
        for pair in pairs:
            identical.setdefault(pair, []).append(record)
        for index, left in enumerate(measures):
            for right in measures[index + 1:]:
                pair = tuple(sorted((left, right)))
                if pair not in pairs:
                    apart.add(pair)
    lessons: dict[str, LearnedLesson] = {}
    for pair, group in identical.items():
        if pair in apart:
            continue
        contexts = {
            tuple(sorted(str(c) for c in (r.observation.get("filter_columns") or ())))
            + (r.run_fingerprint or "",)
            for r in group
        }
        if len(contexts) < 2:
            continue
        status = "active" if len(contexts) >= 3 else "candidate"
        subject = f"{pair[0]} = {pair[1]}"
        lesson_id = _lesson_id("metric_equivalence", source_id, subject)
        lessons[lesson_id] = LearnedLesson(
            lesson_id=lesson_id,
            kind="metric_equivalence",
            subject=subject,
            structured_rule={
                "measures": list(pair),
                "observed": "identical_across_filtered_contexts",
                "contexts": len(contexts),
                "caveat": "reproducible_identity_not_definition",
            },
            evidence_ids=tuple(sorted(r.evidence_id for r in group)),
            confidence="medium" if status == "active" else "low",
            status=status,
            source_dependencies=(source_id,),
            source_fingerprints=fingerprints,
            basis=("reproducible_identity",),
        )
    return lessons


def _preferred_strategy_lessons(
    source_id: str,
    fingerprints: Mapping[str, str],
    records: Sequence[EvidenceRecord],
) -> dict[str, LearnedLesson]:
    """Coarse grain first, candidates restricted, then the drill-down.

    Only verified runs count: a strategy that produced an answer that
    failed verification or the integrity screen proves nothing.
    """
    supporting: list[tuple[EvidenceRecord, list[str], list[str]]] = []
    for record in records:
        if record.observation_type != "strategy_sequence" or not record.trusted:
            continue
        steps = record.observation.get("steps") or ()
        grains = [
            (list(step.get("grain") or []), bool(step.get("filtered")))
            for step in steps
            if isinstance(step, Mapping)
        ]
        found = None
        for i, (coarse, _f) in enumerate(grains):
            for fine, filtered in grains[i + 1:]:
                if coarse and set(coarse) < set(fine) and filtered:
                    found = (coarse, fine)
                    break
            if found:
                break
        if found:
            supporting.append((record, found[0], found[1]))
    if not supporting:
        return {}
    runs = {record.run_fingerprint for record, _c, _f in supporting}
    status = "active" if len(runs) >= 2 else "candidate"
    coarse, fine = supporting[-1][1], supporting[-1][2]
    subject = "coarse to candidate drilldown"
    lesson_id = _lesson_id("preferred_strategy", source_id, subject)
    return {
        lesson_id: LearnedLesson(
            lesson_id=lesson_id,
            kind="preferred_strategy",
            subject=subject,
            structured_rule={
                "strategy": "coarse_to_candidate_drilldown",
                "coarse_grain": coarse,
                "drilldown_grain": fine,
                "verified_runs": len(runs),
                "advice": "restrict_to_candidate_tuples_before_drilldown",
            },
            evidence_ids=tuple(sorted(r.evidence_id for r, _c, _f in supporting)),
            confidence="high" if status == "active" else "medium",
            status=status,
            source_dependencies=(source_id,),
            source_fingerprints=fingerprints,
            basis=("verified_runs",),
        )
    }


def _invalid_path_lessons(
    source_id: str,
    fingerprints: Mapping[str, str],
    records: Sequence[EvidenceRecord],
) -> dict[str, LearnedLesson]:
    by_reference: dict[tuple[str, str], list[EvidenceRecord]] = {}
    for record in records:
        if record.observation_type != "query_execution":
            continue
        reference = record.observation.get("invalid_reference")
        kind = record.observation.get("invalid_reference_kind")
        if isinstance(reference, str) and reference and isinstance(kind, str):
            by_reference.setdefault((reference, kind), []).append(record)
    lessons: dict[str, LearnedLesson] = {}
    for (reference, kind), group in by_reference.items():
        runs = {r.run_fingerprint for r in group}
        status = "active" if len(runs) >= 2 else "candidate"
        subject = f"{kind} {reference}"
        lesson_id = _lesson_id("invalid_path", source_id, subject)
        lessons[lesson_id] = LearnedLesson(
            lesson_id=lesson_id,
            kind="invalid_path",
            subject=subject,
            structured_rule={
                "reference": reference,
                "reference_kind": kind,
                "error": "unknown_reference",
                "occurrences": len(group),
                "runs": len(runs),
            },
            evidence_ids=tuple(sorted(r.evidence_id for r in group)),
            confidence="high" if status == "active" else "low",
            status=status,
            source_dependencies=(source_id,),
            source_fingerprints=fingerprints,
            basis=("catalog_validation",),
        )
    return lessons


def derive_lessons(
    package: KnowledgePackage,
    evidence: Iterable[EvidenceRecord] | None = None,
) -> tuple[LearnedLesson, ...]:
    """Every lesson the package's sources and evidence support right now.

    Structural lessons first, then evidence lessons per source. An
    evidence lesson with the same identity as a structural candidate
    (a name-pattern context requirement confirmed by observation) replaces
    it. The result is not yet merged with the package's existing lessons;
    :func:`promote_lessons` does that and records the transitions.
    """
    records = _current_evidence(package, list(package.evidence) + list(evidence or ()))
    lessons: dict[str, LearnedLesson] = {
        lesson.lesson_id: lesson for lesson in structural_lessons(package)
    }
    by_source: dict[str, list[EvidenceRecord]] = {}
    for record in records:
        for source_id in record.source_ids:
            by_source.setdefault(source_id, []).append(record)
    fingerprints_by_source = {
        source.source_id: {source.source_id: source.schema_fingerprint}
        for source in package.sources
    }
    for source_id, group in sorted(by_source.items()):
        fingerprints = fingerprints_by_source[source_id]
        expensive = _expensive_grain_lessons(source_id, fingerprints, group)
        expensive_grains = {
            tuple(lesson.structured_rule["grain"]) for lesson in expensive.values()
        }
        for derived in (
            expensive,
            _valid_grain_lessons(source_id, fingerprints, group, expensive_grains),
            _context_requirement_lessons(source_id, fingerprints, group),
            _metric_equivalence_lessons(source_id, fingerprints, group),
            _preferred_strategy_lessons(source_id, fingerprints, group),
            _invalid_path_lessons(source_id, fingerprints, group),
        ):
            lessons.update(derived)
    return tuple(lessons[key] for key in sorted(lessons))


def _transition_event(lesson: LearnedLesson, previous: str | None) -> KnowledgeEvent:
    suffix = _domain_fingerprint(
        "fabric-rlm.knowledge.lesson-event.v1",
        {"lesson_id": lesson.lesson_id, "from": previous, "to": lesson.status},
    )[:16]
    return KnowledgeEvent(
        event_id=f"lesson.{lesson.lesson_id}.{lesson.status}.{suffix}",
        event_type=f"lesson.{lesson.status}",
        subject_type="lesson",
        subject_id=lesson.lesson_id,
        status=lesson.status,
        reason_code=lesson.reason_code or ("promoted" if previous else "derived"),
    )


def promote_lessons(
    package: KnowledgePackage,
    evidence: Iterable[EvidenceRecord] = (),
    *,
    max_evidence: int = 5_000,
) -> KnowledgePackage:
    """A new package with the evidence appended and lessons re-derived.

    Existing lessons are replaced by their re-derivation when their
    identity matches; quarantined and retired lessons keep that status
    whatever the evidence says (a person put them there); a lesson the
    evidence no longer supports is retained as it was. Every status change
    is recorded as a lesson event.
    """
    known_ids = {record.evidence_id for record in package.evidence}
    fresh = [
        record
        for record in evidence
        if isinstance(record, EvidenceRecord) and record.evidence_id not in known_ids
    ]
    merged_evidence = list(package.evidence) + fresh
    if len(merged_evidence) > max_evidence:
        merged_evidence = merged_evidence[-max_evidence:]
    source_ids = {source.source_id for source in package.sources}
    merged_evidence = [
        record for record in merged_evidence if set(record.source_ids) <= source_ids
    ]
    staged = KnowledgePackage(
        package_id=package.package_id,
        sources=package.sources,
        relationships=package.relationships,
        operations=package.operations,
        events=package.events,
        evidence=tuple(merged_evidence),
        lessons=package.lessons,
    )
    derived = {lesson.lesson_id: lesson for lesson in derive_lessons(staged)}
    existing = {lesson.lesson_id: lesson for lesson in package.lessons}
    events: list[KnowledgeEvent] = list(package.events)
    event_ids = {event.event_id for event in events}
    final: dict[str, LearnedLesson] = {}
    for lesson_id in sorted(set(existing) | set(derived)):
        before = existing.get(lesson_id)
        after = derived.get(lesson_id)
        if after is None:
            final[lesson_id] = before  # type: ignore[assignment]
            continue
        if before is not None and before.status in {"quarantined", "retired"}:
            after = replace(after, status=before.status, reason_code=before.reason_code)
        final[lesson_id] = after
        previous = before.status if before is not None else None
        if previous != after.status:
            event = _transition_event(after, previous)
            if event.event_id not in event_ids:
                events.append(event)
                event_ids.add(event.event_id)
    return KnowledgePackage(
        package_id=package.package_id,
        sources=package.sources,
        relationships=package.relationships,
        operations=package.operations,
        events=tuple(events),
        evidence=tuple(merged_evidence),
        lessons=tuple(final[key] for key in sorted(final)),
    )


__all__ = ["derive_lessons", "promote_lessons", "structural_lessons"]
