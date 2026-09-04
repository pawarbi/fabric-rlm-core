"""Evidence and lesson contracts, promotion policy, retrieval and staleness."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from fabric_rlm.knowledge import (
    EvidenceRecord,
    KnowledgeEvent,
    KnowledgePackage,
    LearnedLesson,
    SourceProfile,
    canonical_json,
)
from fabric_rlm.knowledge_lessons import (
    derive_lessons,
    promote_lessons,
    structural_lessons,
)
from fabric_rlm.knowledge_preflight import preflight_knowledge
from fabric_rlm.knowledge_retrieval import (
    lesson_score,
    render_learned_guidance,
    retrieve_lessons,
)
from fabric_rlm.knowledge_sources import profile_sources
from fabric_rlm.knowledge_store import (
    read_knowledge_package,
    save_knowledge_package,
)


# -- fixtures -----------------------------------------------------------------


def _model_profile(source_id: str = "arr_model", *, schema_fingerprint: str = "schema-1") -> SourceProfile:
    return SourceProfile(
        source_id=source_id,
        family="semantic_model",
        locator="semantic-model/v1/abc123",
        snapshot_fingerprint="snapshot-1",
        schema_fingerprint=schema_fingerprint,
        schema={
            "tables": {"Period": {"type": "table"}, "ARR Data": {"type": "table"}},
            "columns": {
                "Period[YearQuarter]": {"type": "string"},
                "Period[IsCurrentQuarter]": {"type": "boolean"},
                "Products[Product]": {"type": "string"},
                "Sold To[Region]": {"type": "string"},
                "Sold To[Customer Group]": {"type": "string"},
            },
            "measures": {
                "ARR Data[ARR $]": {"type": "measure"},
                "ARR Data[ARR $ Previous Period]": {"type": "measure"},
                "ARR Data[ARR Growth %]": {"type": "measure"},
                "ARR Data[Active Customers #]": {"type": "measure"},
            },
            "relationships": {},
        },
        diagnostics={"snapshot_exact": True},
        status="active",
    )


def _package(*profiles: SourceProfile, **fields) -> KnowledgePackage:
    return KnowledgePackage(package_id="arr_knowledge", sources=profiles or (_model_profile(),), **fields)


GRAIN_COARSE = ["Products[Product]", "Sold To[Region]"]
GRAIN_FINE = ["Products[Product]", "Sold To[Customer Group]", "Sold To[Region]"]
GRAIN_WIDE = ["Period[YearQuarter]", "Products[Product]", "Sold To[Customer Group]", "Sold To[Region]"]


def _evidence(
    evidence_id: str,
    observation: dict,
    *,
    status: str = "success",
    run: str = "run.one",
    verifier: str = "passed",
    integrity: str = "passed",
    observation_type: str = "query_execution",
    source_id: str = "arr_model",
    fingerprint: str = "schema-1",
    turn: int | None = 1,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        evidence_type="execution" if observation_type == "query_execution" else "trajectory",
        source_ids=(source_id,),
        observation_type=observation_type,
        observation=observation,
        source_fingerprints={source_id: fingerprint},
        execution_status=status,
        verifier_status=verifier,
        analytical_integrity_status=integrity,
        run_fingerprint=run,
        turn=turn,
    )


def _success(evidence_id: str, grain: list[str], *, rows: int = 44, seconds: float = 4.0, measures=("ARR $",), filters: int = 0, run: str = "run.one", **extra) -> EvidenceRecord:
    observation = {
        "query_type": "aggregate",
        "measures": list(measures),
        "measure_count": len(measures),
        "grain": sorted(grain),
        "groupby_count": len(grain),
        "filter_count": filters,
        "returned_rows": rows,
        "total_seconds": seconds,
        "executed": True,
        **extra,
    }
    return _evidence(evidence_id, observation, run=run, **{k: v for k, v in {}.items()})


# -- contracts --------------------------------------------------------------


def test_lessons_hold_structure_not_prose() -> None:
    rule = {"measure": "ARR $ Previous Period", "requires": ["period_context"], "observations": 2}
    lesson = LearnedLesson(
        lesson_id="lesson.context_requirement.abc",
        kind="context_requirement",
        subject="ARR $ Previous Period",
        structured_rule=rule,
        source_dependencies=("arr_model",),
    )
    assert lesson.status == "candidate" and lesson.confidence == "low"
    assert dict(lesson.structured_rule)["requires"] == ("period_context",)
    with pytest.raises(ValueError, match="single line"):
        LearnedLesson(
            lesson_id="lesson.x",
            kind="semantic_fact",
            subject="ARR",
            structured_rule={"note": "ARR Previous Period is tricky.\nUse with care."},
            source_dependencies=("arr_model",),
        )
    with pytest.raises(ValueError, match="at most 256"):
        LearnedLesson(
            lesson_id="lesson.x",
            kind="semantic_fact",
            subject="ARR",
            structured_rule={"note": "x" * 300},
            source_dependencies=("arr_model",),
        )
    with pytest.raises(ValueError, match="kind is not supported"):
        LearnedLesson(lesson_id="lesson.x", kind="opinion", subject="ARR", structured_rule={"a": 1}, source_dependencies=("m",))
    with pytest.raises(ValueError, match="must not be empty"):
        LearnedLesson(lesson_id="lesson.x", kind="semantic_fact", subject="ARR", structured_rule={}, source_dependencies=("m",))
    with pytest.raises(ValueError, match="source_dependencies must not be empty"):
        LearnedLesson(lesson_id="lesson.x", kind="semantic_fact", subject="ARR", structured_rule={"a": 1})
    with pytest.raises(ValueError, match="secret-like"):
        EvidenceRecord(
            evidence_id="evidence.x",
            evidence_type="execution",
            source_ids=("arr_model",),
            observation_type="query_execution",
            observation={"token": "eyJabc.def.ghi"},
        )


def test_evidence_trust_requires_verified_and_integrity_clean() -> None:
    record = _evidence("evidence.a", {"grain": GRAIN_COARSE})
    assert record.trusted
    assert not replace(record, verifier_status="failed").trusted
    assert not replace(record, analytical_integrity_status="unresolved").trusted
    assert not replace(record, execution_status="timeout").trusted
    assert replace(record, analytical_integrity_status="off").trusted
    with pytest.raises(ValueError, match="observation_type is not supported"):
        replace(record, observation_type="gossip")


def test_package_without_learning_records_serializes_exactly_as_version_1() -> None:
    package = _package()
    payload = package.to_dict()
    assert payload["format_version"] == 1
    assert "evidence" not in payload and "lessons" not in payload
    legacy = {key: value for key, value in payload.items()}
    assert KnowledgePackage.from_dict(legacy).fingerprint == package.fingerprint
    with pytest.raises(ValueError, match="carry no learning records"):
        KnowledgePackage.from_dict({**legacy, "lessons": []})


def test_package_with_lessons_round_trips_as_version_2(tmp_path: Path) -> None:
    lesson = structural_lessons(_package())[0]
    evidence = _evidence("evidence.a", {"grain": GRAIN_COARSE, "returned_rows": 3})
    package = _package(evidence=(evidence,), lessons=(lesson,))
    payload = package.to_dict()
    assert payload["format_version"] == 2
    restored = KnowledgePackage.from_dict(json.loads(canonical_json(payload)))
    assert restored == package
    assert restored.fingerprint == package.fingerprint
    destination = tmp_path / "knowledge.json"
    save_knowledge_package(destination, package)
    assert read_knowledge_package(destination) == package
    text = destination.read_text(encoding="utf-8")
    assert "IsCurrentQuarter" in text and "\\n" not in json.dumps(payload["lessons"])


def test_referential_integrity_covers_evidence_and_lessons() -> None:
    lesson = replace(structural_lessons(_package())[0], evidence_ids=("evidence.missing",))
    with pytest.raises(ValueError, match="unknown evidence"):
        _package(lessons=(lesson,))
    with pytest.raises(ValueError, match="unknown source"):
        _package(evidence=(_evidence("evidence.a", {"a": 1}, source_id="other"),))
    with pytest.raises(ValueError, match="unknown source"):
        _package(lessons=(replace(structural_lessons(_package())[0], source_dependencies=("other",)),))
    event = KnowledgeEvent(
        event_id="lesson.x.active",
        event_type="lesson.active",
        subject_type="lesson",
        subject_id="lesson.nope",
        status="active",
    )
    with pytest.raises(ValueError, match="unknown lesson"):
        _package(events=(event,))


def test_persisted_learning_records_reject_free_text_and_credentials(tmp_path: Path) -> None:
    from fabric_rlm.knowledge_store import _parse_envelope

    package = _package(lessons=structural_lessons(_package()))
    envelope = json.loads(json.dumps({
        "format_version": 1,
        "package": package.to_dict(),
        "package_fingerprint": package.fingerprint,
    }))
    envelope["package"]["lessons"][0]["structured_rule"]["api_key"] = "abc"
    with pytest.raises(ValueError, match="privacy-forbidden"):
        _parse_envelope(canonical_json(envelope).encode("utf-8"))
    envelope = json.loads(json.dumps({
        "format_version": 1,
        "package": package.to_dict(),
        "package_fingerprint": package.fingerprint,
    }))
    envelope["package"]["lessons"][0]["structured_rule"]["note"] = "line one\nline two"
    with pytest.raises(ValueError, match="not free text"):
        _parse_envelope(canonical_json(envelope).encode("utf-8"))


# -- structural lessons -------------------------------------------------------


def test_structural_lessons_declare_time_semantics_and_nominate_derived_measures() -> None:
    lessons = structural_lessons(_package())
    by_kind = {}
    for lesson in lessons:
        by_kind.setdefault(lesson.kind, []).append(lesson)
    (time_lesson,) = by_kind["time_semantics"]
    assert time_lesson.status == "active" and time_lesson.confidence == "high"
    assert time_lesson.structured_rule["current_period_constructs"] == ("Period[IsCurrentQuarter]",)
    assert time_lesson.structured_rule["avoid"] == "max_date_inference"
    assert time_lesson.source_fingerprints == {"arr_model": "schema-1"}
    nominated = {lesson.subject: lesson for lesson in by_kind["context_requirement"]}
    assert set(nominated) == {"ARR $ Previous Period", "ARR Growth %"}
    assert all(lesson.status == "candidate" and lesson.confidence == "low" for lesson in nominated.values())
    # a CSV declares nothing
    csv_profile = replace(_model_profile("orders"), family="csv", locator="local/orders")
    assert structural_lessons(KnowledgePackage(package_id="p", sources=(csv_profile,))) == ()


# -- promotion policy ---------------------------------------------------------


def test_expensive_grain_needs_proof_or_repetition() -> None:
    package = _package()
    timeout = _evidence(
        "evidence.t1",
        {"query_type": "aggregate", "grain": GRAIN_WIDE, "reason": "preflight_timeout", "measure_count": 5, "executed": False},
        status="timeout",
        run="run.a",
    )
    lessons = {l.kind: l for l in derive_lessons(package, [timeout])}
    assert lessons["expensive_grain"].status == "candidate"
    assert lessons["expensive_grain"].confidence == "low"

    second = replace(timeout, evidence_id="evidence.t2", run_fingerprint="run.b")
    lessons = {l.kind: l for l in derive_lessons(package, [timeout, second])}
    assert lessons["expensive_grain"].status == "active"
    assert lessons["expensive_grain"].basis == ("repeated_timeout",)

    proof = _evidence(
        "evidence.c1",
        {"query_type": "aggregate", "grain": GRAIN_WIDE, "reason": "cardinality_limit", "estimated_groups": 83000, "max_groups": 10000, "executed": False},
        status="rejected",
    )
    lessons = {l.kind: l for l in derive_lessons(package, [proof])}
    expensive = lessons["expensive_grain"]
    assert expensive.status == "active" and expensive.confidence == "high"
    assert expensive.structured_rule["estimated_groups"] == 83000
    assert expensive.subject == "YearQuarter x Product x Customer Group x Region"
    assert expensive.evidence_ids == ("evidence.c1",)


def test_valid_grain_needs_repeated_verified_success_and_yields_to_failures() -> None:
    package = _package()
    one = _success("evidence.s1", GRAIN_COARSE)
    lessons = {l.kind: l for l in derive_lessons(package, [one])}
    assert lessons["valid_grain"].status == "candidate"

    two = _success("evidence.s2", GRAIN_COARSE, run="run.two", rows=40, seconds=2.5)
    lessons = {l.kind: l for l in derive_lessons(package, [one, two])}
    valid = lessons["valid_grain"]
    assert valid.status == "active" and valid.confidence == "high"
    assert valid.structured_rule["max_rows_observed"] == 44
    assert valid.structured_rule["max_seconds_observed"] == 4.0

    # a run whose answer failed the integrity screen proves nothing
    unverified = [
        replace(one, analytical_integrity_status="unresolved"),
        replace(two, analytical_integrity_status="unresolved"),
    ]
    lessons = {l.kind: l for l in derive_lessons(package, unverified)}
    assert lessons["valid_grain"].status == "candidate"
    assert lessons["valid_grain"].basis == ("single_success",)

    # a timeout at the same grain quarantines the success claim
    timeout = _evidence(
        "evidence.t1",
        {"query_type": "aggregate", "grain": GRAIN_COARSE, "reason": "preflight_timeout", "executed": False},
        status="timeout",
    )
    lessons = {l.kind: l for l in derive_lessons(package, [one, two, timeout])}
    assert lessons["valid_grain"].status == "quarantined"
    assert lessons["valid_grain"].reason_code == "contradicting_evidence"
    assert lessons["expensive_grain"].status == "candidate"


def test_context_requirement_is_confirmed_by_contrast_not_by_name() -> None:
    package = _package()
    nominated = {l.subject: l for l in derive_lessons(package)}["ARR $ Previous Period"]
    assert nominated.status == "candidate" and nominated.confidence == "low"

    degenerate = _success(
        "evidence.d1",
        [],
        measures=("ARR $", "ARR $ Previous Period", "ARR Growth %"),
        rows=1,
        measure_identities=[["ARR $", "ARR $ Previous Period"]],
        constant_measures={"ARR Growth %": "zero"},
    )
    lessons = {l.subject: l for l in derive_lessons(package, [degenerate])}
    previous = lessons["ARR $ Previous Period"]
    assert previous.status == "candidate" and previous.confidence == "medium"
    assert previous.structured_rule["observed"] == "identity_under_unfiltered_context"
    assert previous.structured_rule["base_measure"] == "ARR $"
    growth = lessons["ARR Growth %"]
    assert growth.structured_rule["observed"] == "constant_under_unfiltered_context"
    assert growth.structured_rule["constant"] == "zero"

    contrast = _success(
        "evidence.d2",
        ["Period[YearQuarter]"],
        measures=("ARR $", "ARR $ Previous Period"),
        rows=10,
        filters=1,
        run="run.two",
    )
    lessons = {l.subject: l for l in derive_lessons(package, [degenerate, contrast])}
    previous = lessons["ARR $ Previous Period"]
    assert previous.status == "active" and previous.confidence == "high"
    assert previous.basis == ("degenerate_unfiltered", "distinct_when_filtered")
    assert set(previous.evidence_ids) == {"evidence.d1", "evidence.d2"}

    repeated = replace(degenerate, evidence_id="evidence.d3", run_fingerprint="run.three")
    lessons = {l.subject: l for l in derive_lessons(package, [degenerate, repeated])}
    assert lessons["ARR $ Previous Period"].status == "active"
    assert lessons["ARR $ Previous Period"].confidence == "medium"
    # the unfiltered identity is never metric equivalence
    assert "metric_equivalence" not in {l.kind for l in lessons.values()}


def test_metric_equivalence_needs_identity_across_filtered_contexts() -> None:
    package = _package()
    identical = [
        _success(
            f"evidence.e{i}",
            grain,
            measures=("ARR $", "Active Customers #"),
            filters=1,
            run=f"run.{i}",
            measure_identities=[["ARR $", "Active Customers #"]],
            filter_columns=[column],
        )
        for i, (grain, column) in enumerate(
            [(GRAIN_COARSE, "Period[YearQuarter]"), (GRAIN_COARSE, "Sold To[Region]"), (GRAIN_FINE, "Products[Product]")]
        )
    ]
    two = {l.kind: l for l in derive_lessons(package, identical[:2])}
    assert two["metric_equivalence"].status == "candidate"
    three = {l.kind: l for l in derive_lessons(package, identical)}
    assert three["metric_equivalence"].status == "active"
    assert three["metric_equivalence"].structured_rule["caveat"] == "reproducible_identity_not_definition"
    apart = _success("evidence.apart", GRAIN_COARSE, measures=("ARR $", "Active Customers #"), filters=1, run="run.x")
    assert "metric_equivalence" not in {l.kind for l in derive_lessons(package, identical + [apart])}


def test_preferred_strategy_comes_only_from_verified_runs() -> None:
    package = _package()
    sequence = _evidence(
        "evidence.seq1",
        {"steps": [{"grain": sorted(GRAIN_COARSE), "filtered": False}, {"grain": sorted(GRAIN_FINE), "filtered": True}], "step_count": 2},
        observation_type="strategy_sequence",
        run="run.a",
        turn=None,
    )
    lessons = {l.kind: l for l in derive_lessons(package, [sequence])}
    strategy = lessons["preferred_strategy"]
    assert strategy.status == "candidate" and strategy.confidence == "medium"
    assert strategy.structured_rule["strategy"] == "coarse_to_candidate_drilldown"
    second = replace(sequence, evidence_id="evidence.seq2", run_fingerprint="run.b")
    lessons = {l.kind: l for l in derive_lessons(package, [sequence, second])}
    assert lessons["preferred_strategy"].status == "active"
    # the bad cold answer: same sequence, integrity unresolved; teaches nothing
    bad = replace(sequence, evidence_id="evidence.bad", analytical_integrity_status="unresolved", run_fingerprint="run.bad")
    assert "preferred_strategy" not in {l.kind for l in derive_lessons(package, [bad])}
    # but its typed timeout still teaches
    timeout = _evidence(
        "evidence.badt",
        {"query_type": "aggregate", "grain": GRAIN_WIDE, "reason": "cardinality_limit", "estimated_groups": 50000, "max_groups": 10000, "executed": False},
        status="rejected",
        integrity="unresolved",
        run="run.bad",
    )
    kinds = {l.kind for l in derive_lessons(package, [bad, timeout])}
    assert "expensive_grain" in kinds and "preferred_strategy" not in kinds


def test_invalid_path_is_promoted_only_when_repeated_across_runs() -> None:
    package = _package()
    rejected = _evidence(
        "evidence.r1",
        {"query_type": "aggregate", "reason": "validation", "invalid_reference": "ARR Growth", "invalid_reference_kind": "measure", "error_class": "SemanticModelQueryError"},
        status="rejected",
        run="run.a",
    )
    lessons = {l.kind: l for l in derive_lessons(package, [rejected])}
    assert lessons["invalid_path"].status == "candidate"
    again = replace(rejected, evidence_id="evidence.r2", run_fingerprint="run.b")
    lessons = {l.kind: l for l in derive_lessons(package, [rejected, again])}
    assert lessons["invalid_path"].status == "active"
    assert lessons["invalid_path"].structured_rule["reference"] == "ARR Growth"


def test_promotion_records_transitions_keeps_quarantine_and_ignores_stale_evidence() -> None:
    package = _package()
    timeout = _evidence(
        "evidence.t1",
        {"query_type": "aggregate", "grain": GRAIN_WIDE, "reason": "preflight_timeout", "executed": False},
        status="timeout",
        run="run.a",
    )
    first = promote_lessons(package, [timeout])
    expensive = next(l for l in first.lessons if l.kind == "expensive_grain")
    assert expensive.status == "candidate"
    assert first.format_version == 2 and len(first.evidence) == 1
    kinds = {(e.subject_type, e.status) for e in first.events}
    assert ("lesson", "candidate") in kinds and ("lesson", "active") in kinds

    quarantined = replace(first, lessons=tuple(
        replace(l, status="quarantined", reason_code="reviewed") if l.lesson_id == expensive.lesson_id else l
        for l in first.lessons
    ))
    second = promote_lessons(quarantined, [replace(timeout, evidence_id="evidence.t2", run_fingerprint="run.b")])
    kept = next(l for l in second.lessons if l.lesson_id == expensive.lesson_id)
    assert kept.status == "quarantined" and kept.reason_code == "reviewed"
    assert len(second.evidence) == 2
    # the same evidence again is not appended twice
    assert len(promote_lessons(second, [timeout]).evidence) == 2

    stale = replace(timeout, evidence_id="evidence.old", run_fingerprint="run.old", source_fingerprints={"arr_model": "schema-0"})
    third = promote_lessons(package, [stale])
    assert "expensive_grain" not in {l.kind for l in third.lessons}
    assert len(third.evidence) == 1


# -- retrieval and rendering --------------------------------------------------


def _learned_package() -> KnowledgePackage:
    package = _package()
    proof = _evidence(
        "evidence.c1",
        {"query_type": "aggregate", "grain": GRAIN_WIDE, "reason": "cardinality_limit", "estimated_groups": 83000, "max_groups": 10000, "executed": False},
        status="rejected",
    )
    degenerate = _success(
        "evidence.d1", [], measures=("ARR $", "ARR $ Previous Period"), rows=1,
        measure_identities=[["ARR $", "ARR $ Previous Period"]],
    )
    contrast = _success(
        "evidence.d2", ["Period[YearQuarter]"], measures=("ARR $", "ARR $ Previous Period"), rows=10, filters=1, run="run.two",
    )
    coarse = [_success(f"evidence.s{i}", GRAIN_COARSE, run=f"run.{i}") for i in range(2)]
    sequences = [
        _evidence(
            f"evidence.seq{i}",
            {"steps": [{"grain": sorted(GRAIN_COARSE), "filtered": False}, {"grain": sorted(GRAIN_FINE), "filtered": True}], "step_count": 2},
            observation_type="strategy_sequence", run=f"run.{i}", turn=None,
        )
        for i in range(2)
    ]
    return promote_lessons(package, [proof, degenerate, contrast, *coarse, *sequences])


def test_retrieval_is_scoped_to_the_task_and_never_shows_candidates() -> None:
    package = _learned_package()
    active = {l.kind for l in package.lessons if l.status == "active"}
    assert active == {"time_semantics", "context_requirement", "expensive_grain", "valid_grain", "preferred_strategy"}

    time_task = retrieve_lessons(package, "What is total ARR for the current quarter?")
    assert [l.kind for l in time_task][:1] == ["time_semantics"]
    assert "preferred_strategy" not in {l.kind for l in time_task}

    trend = retrieve_lessons(
        package,
        "Identify the product x region x customer group segments whose ARR deteriorated over the last three quarters and rank them",
    )
    kinds = [l.kind for l in trend]
    assert "expensive_grain" in kinds and "preferred_strategy" in kinds and "context_requirement" in kinds
    assert retrieve_lessons(package, "How many rows does the file have?") == ()
    assert len(retrieve_lessons(package, "current quarter ARR by product and region trend growth", limit=2)) == 2
    candidates = retrieve_lessons(package, "ARR Growth % this quarter", statuses=("candidate",))
    assert all(l.status == "candidate" for l in candidates)
    with pytest.raises(ValueError):
        retrieve_lessons(package, "x", limit=-1)


def test_rendering_states_rules_and_confidence_without_values() -> None:
    package = _learned_package()
    lessons = retrieve_lessons(package, "ARR growth by product and region for the current quarter")
    text = render_learned_guidance(lessons)
    assert text.startswith("## Learned source guidance")
    assert "Period[IsCurrentQuarter]" in text and "MAX(Date)" in text
    assert "ARR $ Previous Period requires an explicit period context" in text
    assert "83,000 groups, limit 10,000" in text
    assert "restrict to the candidate tuples" in text
    assert "Confidence:" in text and "source declared" in text
    assert "926" not in text and "run." not in text and "evidence." not in text
    assert render_learned_guidance(()) == ""
    assert lesson_score(lessons[0], set()) == 0.0


# -- staleness ----------------------------------------------------------------


def test_schema_drift_stales_only_dependent_lessons(tmp_path: Path) -> None:
    orders = tmp_path / "orders.csv"
    orders.write_text("order_id,amount\n1,10\n", encoding="utf-8")
    customers = tmp_path / "customers.csv"
    customers.write_text("customer_id,region\n1,west\n", encoding="utf-8")
    sources = {"orders": orders, "customers": customers}
    profiles = tuple(replace(p, status="active") for p in profile_sources(sources))
    package = KnowledgePackage(package_id="p", sources=profiles)
    lessons = tuple(
        LearnedLesson(
            lesson_id=f"lesson.valid_grain.{source_id}",
            kind="valid_grain",
            subject=f"{source_id} grain",
            structured_rule={"grain": ["region"], "advice": "reliable_analysis_grain"},
            status="active",
            confidence="medium",
            source_dependencies=(source_id,),
            source_fingerprints={source_id: profile.schema_fingerprint},
        )
        for source_id, profile in zip(("orders", "customers"), profiles)
    )
    package = replace(package, lessons=lessons)

    customers.write_text("customer_id,region,tier\n1,west,gold\n", encoding="utf-8")
    result = preflight_knowledge(package, sources)
    assert result.drift == {"customers": "schema"}
    statuses = {l.lesson_id: l.status for l in result.package.lessons}
    assert statuses == {"lesson.valid_grain.orders": "active", "lesson.valid_grain.customers": "stale"}
    assert any(e.subject_type == "lesson" and e.subject_id == "lesson.valid_grain.customers" for e in result.package.events)

    orders.write_text("order_id,amount\n1,10\n2,20\n", encoding="utf-8")
    customers.write_text("customer_id,region\n1,west\n", encoding="utf-8")
    result = preflight_knowledge(package, sources)
    assert result.drift == {"orders": "snapshot"}
    assert {l.status for l in result.package.lessons} == {"active"}
