"""Lesson derivation and promotion: evidence in, structured lessons out.

Two kinds of lesson exist. Structural lessons come from what a source's
schema suggests about itself (a column named like a current-period flag)
and are active on arrival, labelled as name-inferred. Evidence lessons are
promoted from :class:`~fabric_rlm.knowledge.EvidenceRecord` values by a
per-kind policy whose bar is the strength of the evidence, never its
repetition alone:

* a query rejected by the cardinality preflight proves an expensive grain at
  once; one timeout only nominates it;
* a grain that executed and returned rows twice is a valid grain (an
  execution fact); its confidence rises with analytically verified runs;
* a derived measure identical to its base under an unfiltered context is a
  candidate context requirement, and stays one however often it recurs,
  because a flat business produces the same identity; only the same pair
  observed distinct under a period filter confirms it;
* two measures identical across filtered contexts are an observed identity
  (``query_behavior``), never a semantic equivalence: equal values do not
  establish equal definitions;
* a strategy is preferred only when the trajectory proved it (a candidate
  restriction between the coarse and the fine query) in runs whose answer
  passed a verifier and the integrity screen.

Lessons carry the evidence they rest on, the fingerprints they depend on
and a dependency scope (schema, snapshot or operational) that decides what
kind of drift stales them. Quarantined and retired lessons stay that way.
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


# "IsCurrentQuarter", "CurrentYearQuarter", "current_period", "AsOfDate",
# "LatestPeriod"; not "concurrent" or "currently".
_CURRENT_PERIOD = re.compile(
    r"(?i)(?:^|[^a-z])(?:is[_ ]?)?current(?-i:(?![a-z]{2,}))"
    r"|(?:^|[^a-z])as[_ ]?of(?:$|[^a-z]|(?-i:(?=[A-Z])))"
    r"|(?:^|[^a-z])latest[_ ]?(?:period|quarter|month|date|week)"
)
_DERIVED_TIME_MEASURE = re.compile(
    r"(?i)previous|prior|(?:^|[^a-z])py(?:$|[^a-z])|(?:^|[^a-z])pp(?:$|[^a-z])|yoy|qoq|mom|growth|"
    r"(?:^|[^a-z])nrr(?:$|[^a-z])|(?:^|[^a-z])grr(?:$|[^a-z])|retention|churn|(?:^|[^a-z])change|"
    r"delta|variance|(?:^|[^a-z])vs(?:$|[^a-z])|last (?:year|quarter|month|period)|ttm|ltm|ytd|qtd|mtd"
)
_PERIOD_TABLE = re.compile(r"(?i)period|date|calendar|time|fiscal")
_MAX_CONSTRUCTS = 10
_MAX_MEASURES_PER_GRAIN = 10
_MAX_CONTEXT_CANDIDATES = 40


def _leaf(name: str) -> str:
    """``Period[IsCurrentQuarter]`` gives ``IsCurrentQuarter``."""
    text = str(name).strip()
    if text.endswith("]") and "[" in text:
        return text[text.rindex("[") + 1:-1].strip() or text
    return text


def _table_of(name: str) -> str:
    text = str(name).strip()
    return text[: text.index("[")].strip() if "[" in text else ""


def _lesson_id(kind: str, source_id: str, subject: str) -> str:
    suffix = _domain_fingerprint(
        "fabric-rlm.knowledge.lesson.v1",
        {"kind": kind, "source_id": source_id, "subject": subject.strip().lower()},
    )[:16]
    return f"lesson.{kind}.{suffix}"


def _grain_subject(grain: Sequence[str]) -> str:
    return " x ".join(_leaf(item) for item in grain) or "total"


def _schema_section(profile: SourceProfile, family: str) -> Mapping[str, object]:
    section = profile.schema.get(family) if isinstance(profile.schema, Mapping) else None
    return section if isinstance(section, Mapping) else {}


def structural_lessons(package: KnowledgePackage) -> tuple[LearnedLesson, ...]:
    """Lessons a source's schema suggests about itself, active on arrival.

    A semantic model with a column or measure named like a current-period
    construct (``Period[IsCurrentQuarter]``, ``Calendar[AsOfDate]``) most
    likely defines "current" itself, and an agent must not infer it from the
    maximum date. The source did not declare that role; the library read it
    from the name, so the lesson is labelled ``schema_name_pattern`` with
    medium confidence, high only when the construct is a boolean column in a
    period-like table. Measures named as time-relative or derived are
    nominated, as candidates only, for a context requirement that evidence
    can later confirm.
    """
    lessons: list[LearnedLesson] = []
    for profile in package.sources:
        if profile.family != "semantic_model":
            continue
        fingerprints = {profile.source_id: profile.schema_fingerprint}
        columns = _schema_section(profile, "columns")
        measures = _schema_section(profile, "measures")
        # Only names inside a period-like table count at all: a
        # CurrentBalance on an accounts table says nothing about time. A
        # boolean flag there is the strong form and goes active; a name
        # match alone is a candidate that a probe can later confirm.
        constructs = [
            str(name)
            for name in list(columns) + list(measures)
            if _CURRENT_PERIOD.search(_leaf(str(name))) and _PERIOD_TABLE.search(_table_of(str(name)))
        ]
        flags = [
            name
            for name in constructs
            if name in columns
            and isinstance(columns.get(name), Mapping)
            and str(columns[name].get("type", "")).lower() in {"boolean", "bool"}
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
                        "inferred_from": "schema_names",
                    },
                    confidence="high" if flags else "medium",
                    status="active" if flags else "candidate",
                    source_dependencies=(profile.source_id,),
                    source_fingerprints=fingerprints,
                    basis=("schema_name_pattern", "boolean_period_flag") if flags else ("schema_name_pattern",),
                    dependency_scope="schema",
                )
            )
        derived = [str(name) for name in measures if _DERIVED_TIME_MEASURE.search(_leaf(str(name)))]
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
                    dependency_scope="schema",
                )
            )
    return tuple(lessons)


def _current_evidence(
    package: KnowledgePackage,
    evidence: Iterable[EvidenceRecord],
) -> list[EvidenceRecord]:
    """Evidence whose stamped schema fingerprints match the package's sources.

    A record captured against an older schema is not thrown away, but it
    does not promote anything against the schema that replaced it. A record
    with no fingerprint for one of its sources is unknown, not current: it
    promotes nothing either.
    """
    current = {source.source_id: source.schema_fingerprint for source in package.sources}
    kept: list[EvidenceRecord] = []
    for record in evidence:
        if any(source_id not in current for source_id in record.source_ids):
            continue
        if all(
            record.source_fingerprints.get(source_id) == current[source_id]
            for source_id in record.source_ids
        ):
            kept.append(record)
    return kept


def current_evidence(
    package: KnowledgePackage,
    evidence: Iterable[EvidenceRecord],
) -> list[EvidenceRecord]:
    """The records in ``evidence`` whose fingerprints match ``package``."""
    return _current_evidence(package, evidence)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _pairs(value: Any) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for pair in value or ():
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            pairs.add(tuple(sorted((str(pair[0]), str(pair[1])))))  # type: ignore[arg-type]
    return pairs


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
        estimates = [e for e in (_number(r.observation.get("estimated_groups")) for r in cardinality) if e is not None]
        limits = [l for l in (_number(r.observation.get("max_groups")) for r in group) if l is not None]
        # Repetition means independent runs: a retry inside one run is the
        # same observation, not a second one.
        timeout_runs = {r.run_fingerprint for r in timeouts}
        if cardinality:
            status, confidence, basis = "active", "high", ("preflight_estimate",)
            outcome = "cardinality_limit"
        elif len(timeout_runs) >= 2:
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
        measure_counts = [m for m in (_number(r.observation.get("measure_count")) for r in group) if m is not None]
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
            dependency_scope="snapshot",
        )
    return lessons


def _valid_grain_lessons(
    source_id: str,
    fingerprints: Mapping[str, str],
    records: Sequence[EvidenceRecord],
    expensive_grains: set[tuple[str, ...]],
) -> dict[str, LearnedLesson]:
    """A grain that executed and returned rows, twice: an execution fact.

    The answer built on the query does not have to have been verified for
    the query to have run; verified runs only raise the confidence.
    """
    by_grain: dict[tuple[str, ...], list[EvidenceRecord]] = {}
    for record in records:
        if record.observation_type != "query_execution" or not record.execution_trusted:
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
        verified = [r for r in group if r.analytically_trusted]
        runs = {r.run_fingerprint for r in group}
        verified_runs = {r.run_fingerprint for r in verified}
        if len(runs) >= 2:
            status = "active"
            confidence = "high" if len(verified_runs) >= 2 else "medium"
        else:
            status, confidence = "candidate", "low"
        basis = ["repeated_execution" if len(runs) >= 2 else "single_execution"]
        if verified:
            basis.append("verified_runs")
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
            "runs": len(runs),
            "verified_successes": len(verified),
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
            dependency_scope="snapshot",
        )
    return lessons


def _is_derived_measure(name: str) -> bool:
    return bool(_DERIVED_TIME_MEASURE.search(_leaf(name)))


_PERIOD_COLUMN = re.compile(r"(?i)period|quarter|month|year|date|week|fiscal|day")


def _is_period_column(column: object) -> bool:
    return bool(
        _PERIOD_TABLE.search(_table_of(str(column))) or _PERIOD_COLUMN.search(_leaf(str(column)))
    )


def _has_period_context(observation: Mapping[str, Any]) -> bool:
    """Whether the query pinned a period, by filter or by grouping.

    Telemetry keeps filter values out, so a period filter is a filter on a
    column whose table or name is period-like; a query grouped by such a
    column pins one period per row just as well. A region filter that
    happens to separate two measures says nothing about period context.
    """
    return any(
        _is_period_column(column)
        for column in list(observation.get("filter_columns") or ()) + list(observation.get("grain") or ())
    )


def _context_requirement_lessons(
    source_id: str,
    fingerprints: Mapping[str, str],
    records: Sequence[EvidenceRecord],
) -> dict[str, LearnedLesson]:
    """A derived measure that collapses without a period context.

    Under an unfiltered context ``ARR $ Previous Period`` returned exactly
    ``ARR $`` (or a growth rate a constant zero). That nominates the lesson
    and nothing more: a business that is flat produces the same identity,
    however many runs observe it. The lesson becomes active only when the
    same pair was compared again under a filter on a period dimension and
    came out distinct, or a constant derived measure came out non-constant
    there. A contrast under some other filter (a region, a product) is not
    period evidence and does not count.
    """
    degenerate: dict[str, list[tuple[EvidenceRecord, str | None, str | None]]] = {}
    filtered_distinct: dict[tuple[str, str], list[EvidenceRecord]] = {}
    filtered_varying: dict[str, list[EvidenceRecord]] = {}
    for record in records:
        if record.observation_type != "query_execution" or not record.execution_trusted:
            continue
        observation = record.observation
        measures = [m for m in (observation.get("measures") or ()) if isinstance(m, str)]
        period_context = _has_period_context(observation)
        unfiltered = not observation.get("filter_count") and not period_context
        identities = _pairs(observation.get("measure_identities"))
        compared = _pairs(observation.get("compared_measure_pairs")) or identities
        constants = observation.get("constant_measures") or {}
        if unfiltered:
            for left, right in identities:
                derived, base = (left, right) if _is_derived_measure(left) and not _is_derived_measure(right) else (
                    (right, left) if _is_derived_measure(right) and not _is_derived_measure(left) else (None, None)
                )
                if derived:
                    degenerate.setdefault(derived, []).append((record, base, None))
            for measure, constant in constants.items():
                if _is_derived_measure(str(measure)) and constant in {"zero", "one"}:
                    degenerate.setdefault(str(measure), []).append((record, None, str(constant)))
        elif period_context:
            for pair in compared:
                if pair not in identities:
                    filtered_distinct.setdefault(pair, []).append(record)
            for measure in measures:
                if _is_derived_measure(measure) and constants.get(measure) not in {"zero", "one"}:
                    filtered_varying.setdefault(measure, []).append(record)
    lessons: dict[str, LearnedLesson] = {}
    for measure, observations in degenerate.items():
        bases = sorted({b for _r, b, _c in observations if b})
        constant_codes = sorted({c for _r, _b, c in observations if c})
        contrasts: list[EvidenceRecord] = []
        for base in bases:
            contrasts.extend(filtered_distinct.get(tuple(sorted((measure, base))), []))  # type: ignore[arg-type]
        if constant_codes and not bases:
            contrasts.extend(filtered_varying.get(measure, []))
        runs = {r.run_fingerprint for r, _b, _c in observations}
        if contrasts:
            status, confidence, basis = "active", "high", ("degenerate_unfiltered", "distinct_when_filtered")
        elif len(runs) >= 2:
            status, confidence, basis = "candidate", "medium", ("degenerate_unfiltered", "repeated_runs")
        else:
            status, confidence, basis = "candidate", "medium", ("degenerate_unfiltered",)
        rule: dict[str, Any] = {
            "measure": measure,
            "requires": ["period_context"],
            "observed": "identity_under_unfiltered_context" if bases else "constant_under_unfiltered_context",
            "fallback_strategy": "explicit_period_base_measure",
            "observations": len(observations),
            "contrasting_observations": len(contrasts),
        }
        if bases:
            rule["base_measure"] = bases[0]
        if constant_codes:
            rule["constant"] = constant_codes[0]
        lesson_id = _lesson_id("context_requirement", source_id, measure)
        evidence_ids = sorted({r.evidence_id for r, _b, _c in observations} | {r.evidence_id for r in contrasts})
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
            dependency_scope="schema",
        )
    return lessons


def _observed_identity_lessons(
    source_id: str,
    fingerprints: Mapping[str, str],
    records: Sequence[EvidenceRecord],
) -> dict[str, LearnedLesson]:
    """Two measures identical across several filtered contexts, never apart.

    Recorded as ``query_behavior``, an observed identity with
    ``semantic_equivalence`` false: values that coincide in the observed
    slices say nothing about definitions, units, populations or
    aggregation semantics. A ``metric_equivalence`` lesson needs structural
    evidence this passive path does not produce. An identity under an
    unfiltered context does not count at all: that is the signature of a
    missing context.
    """
    identical: dict[tuple[str, str], list[EvidenceRecord]] = {}
    apart: set[tuple[str, str]] = set()
    for record in records:
        if record.observation_type != "query_execution" or not record.execution_trusted:
            continue
        observation = record.observation
        if not observation.get("filter_count"):
            continue
        identities = _pairs(observation.get("measure_identities"))
        compared = _pairs(observation.get("compared_measure_pairs")) or identities
        for pair in identities:
            identical.setdefault(pair, []).append(record)
        apart |= compared - identities
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
        subject = f"{pair[0]} = {pair[1]} observed"
        lesson_id = _lesson_id("query_behavior", source_id, subject)
        lessons[lesson_id] = LearnedLesson(
            lesson_id=lesson_id,
            kind="query_behavior",
            subject=subject,
            structured_rule={
                "measures": list(pair),
                "observation": "identical_in_observed_contexts",
                "contexts": len(contexts),
                "semantic_equivalence": False,
                "caveat": "values_coincide_definitions_unverified",
            },
            evidence_ids=tuple(sorted(r.evidence_id for r in group)),
            confidence="low",
            status=status,
            source_dependencies=(source_id,),
            source_fingerprints=fingerprints,
            basis=("observed_identity",),
            dependency_scope="snapshot",
        )
    return lessons


def _preferred_strategy_lessons(
    source_id: str,
    fingerprints: Mapping[str, str],
    records: Sequence[EvidenceRecord],
) -> dict[str, LearnedLesson]:
    """Coarse grain first, candidates restricted, then the drill-down.

    Only a trajectory that demonstrably preserved the candidate identity
    between the coarse and the fine query counts (the harvester records
    ``candidate_identity_preserved`` when a ``restrict_to_candidate_tuples``
    call sits between them); a finer query that merely carried a filter
    proves nothing. And only runs whose answer passed a verifier and the
    integrity screen count: a strategy that produced an unverified answer
    is not a strategy to prefer.
    """
    supporting: list[tuple[EvidenceRecord, list[str], list[str]]] = []
    for record in records:
        if record.observation_type != "strategy_sequence" or not record.analytically_trusted:
            continue
        observation = record.observation
        if observation.get("strategy") != "candidate_drilldown" or observation.get("candidate_identity_preserved") is not True:
            continue
        coarse = [str(c) for c in (observation.get("candidate_source_grain") or ())]
        fine = [str(c) for c in (observation.get("drilldown_grain") or ())]
        if coarse and fine and set(coarse) < set(fine):
            supporting.append((record, coarse, fine))
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
            basis=("verified_runs", "candidate_restriction_observed"),
            dependency_scope="operational",
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
            dependency_scope="schema",
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
            _observed_identity_lessons(source_id, fingerprints, group),
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


def _evict(
    evidence: Sequence[EvidenceRecord],
    *,
    keep_ids: set[str],
    max_evidence: int,
) -> list[EvidenceRecord]:
    """Drop the oldest unreferenced records until the cap is met.

    A record a retained lesson cites is never evicted, whatever its age, so
    the package's referential integrity survives any number of enrichments.
    """
    if len(evidence) <= max_evidence:
        return list(evidence)
    excess = len(evidence) - max_evidence
    kept: list[EvidenceRecord] = []
    for record in evidence:
        if excess > 0 and record.evidence_id not in keep_ids:
            excess -= 1
            continue
        kept.append(record)
    return kept


def promote_lessons(
    package: KnowledgePackage,
    evidence: Iterable[EvidenceRecord] = (),
    *,
    max_evidence: int = 5_000,
) -> KnowledgePackage:
    """A new package with the evidence appended and lessons re-derived.

    Incoming evidence is deduplicated against the package and within the
    call. Existing lessons are replaced by their re-derivation when their
    identity matches; quarantined and retired lessons keep that status
    whatever the evidence says (a person put them there); a stale lesson
    stays stale until a record from this call, matching the package's
    current fingerprints, supports its re-derivation (the evidence already
    in the package is what went stale, so re-reading it proves nothing); a
    lesson the evidence no longer supports is retained as it was. Every
    status change is recorded as a lesson event.
    """
    known_ids = {record.evidence_id for record in package.evidence}
    fresh: list[EvidenceRecord] = []
    for record in evidence:
        if isinstance(record, EvidenceRecord) and record.evidence_id not in known_ids:
            known_ids.add(record.evidence_id)
            fresh.append(record)
    source_ids = {source.source_id for source in package.sources}
    fresh_current_ids = {record.evidence_id for record in _current_evidence(package, fresh)}
    merged_evidence = [
        record
        for record in list(package.evidence) + fresh
        if set(record.source_ids) <= source_ids
    ]
    referenced = {
        evidence_id
        for lesson in package.lessons
        for evidence_id in lesson.evidence_ids
    }
    merged_evidence = _evict(merged_evidence, keep_ids=referenced, max_evidence=max_evidence)
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
        elif (
            before is not None
            and before.status == "stale"
            and not (set(after.evidence_ids) & fresh_current_ids)
        ):
            after = replace(after, status="stale", reason_code=before.reason_code)
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


__all__ = ["current_evidence", "derive_lessons", "promote_lessons", "structural_lessons"]
