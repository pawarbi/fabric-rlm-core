"""Evidence capture, source-call telemetry, guidance injection and enrich."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from fabric_rlm import RLM, Knowledge, load_knowledge
from fabric_rlm import _worker
from fabric_rlm.interpreter import ExecResult, Interpreter
from fabric_rlm.knowledge import EvidenceRecord, KnowledgePackage, LearnedLesson
from fabric_rlm.knowledge_benchmark import (
    KnowledgeBenchmarkTask,
    run_knowledge_benchmark,
)
from fabric_rlm.knowledge_evidence import (
    harvest_evidence,
    run_fingerprint_for,
    source_call_summary,
)
from fabric_rlm.prompts import build_system_prompt
from fabric_rlm.runtime import RLMResult
from fabric_rlm.trajectory import Trajectory, TurnRecord


class ScriptedLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict[str, str]]] = []

    def __call__(self, *, messages):
        self.messages.append([dict(message) for message in messages])
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def _code(body: str) -> str:
    return f"```python\n{body}\n```"


def _csv(tmp_path: Path) -> Path:
    path = tmp_path / "orders.csv"
    path.write_text("order_id,amount\n1,10.5\n2,20.0\n", encoding="utf-8")
    return path


def _system_prompt(lm: ScriptedLM) -> str:
    """The agent's system prompt; a knowledge run may call the planner first."""
    return next(
        call[0]["content"]
        for call in lm.messages
        if "## Task" in call[0]["content"]
    )


class Telemetered:
    """A namespace object with the query_telemetry shape of SemanticModel."""

    def __init__(self) -> None:
        self.log: list[dict] = []

    @property
    def query_telemetry(self) -> tuple[dict, ...]:
        return tuple(dict(item) for item in self.log)


AGGREGATE_OK = {
    "query_type": "aggregate",
    "executed": True,
    "measures": ["ARR $", "ARR $ Previous Period"],
    "groupby": ["Products[Product]", "Sold To[Region]"],
    "groupby_count": 2,
    "measure_count": 2,
    "filter_columns": [],
    "filter_count": 0,
    "top": None,
    "order_by": None,
    "max_groups": 10000,
    "preflight": True,
    "preflight_seconds": 0.4,
    "estimated_groups": 44,
    "query": "EVALUATE SUMMARIZECOLUMNS(... TREATAS({\"2026/Q2\"} ...)",
    "execution_seconds": 3.1,
    "returned_rows": 44,
    "total_seconds": 3.5,
    "measure_identities": [["ARR $", "ARR $ Previous Period"]],
}
AGGREGATE_TOO_BROAD = {
    "query_type": "aggregate",
    "executed": False,
    "measures": ["ARR $"],
    "groupby": ["Period[YearQuarter]", "Products[Product]", "Sold To[Customer Group]", "Sold To[Region]"],
    "groupby_count": 4,
    "measure_count": 1,
    "filter_columns": [],
    "filter_count": 0,
    "max_groups": 10000,
    "preflight": True,
    "preflight_seconds": 8.2,
    "estimated_groups": 83000,
    "reason": "cardinality_limit",
}
AGGREGATE_UNKNOWN = {
    "query_type": "aggregate",
    "executed": False,
    "reason": "validation",
    "error": "SemanticModelQueryError: Unknown semantic-model measure: ARR Growth\n\nAvailable close matches:\n- ARR Growth %",
}
LAKEHOUSE_OK = {
    "query_type": "lakehouse_sql",
    "source_root": "abfss://ws@onelake/lh/Tables",
    "query_fingerprint": "abcd1234abcd1234",
    "query_chars": 80,
    "source_count": 1,
    "max_rows": 1000,
    "executed": True,
    "execution_seconds": 1.2,
    "returned_rows": 12,
    "truncated": False,
    "total_seconds": 1.2,
}


def _turn(number: int, code: str, *, calls=(), submitted=False, error=None, turn_type="normal") -> TurnRecord:
    return TurnRecord(
        turn=number,
        code=code,
        stdout="",
        stderr="",
        error=error,
        submitted=submitted,
        state={},
        turn_type=turn_type,
        source_calls=[dict(call) for call in calls],
    )


def _result(turns, *, submitted=True, failure_reason=None, metadata=None, payload=None) -> RLMResult:
    trajectory = Trajectory(turns=list(turns), metadata=dict(metadata or {}))
    return RLMResult(
        submitted=submitted,
        payload=(payload or {"answer": "x"}) if submitted else None,
        trajectory=trajectory,
        final_state={},
        failure_reason=failure_reason,
    )


# -- worker and interpreter telemetry -----------------------------------------


def test_worker_reports_new_telemetry_records_once_per_turn() -> None:
    model = Telemetered()
    _worker._namespace.clear()
    _worker._TELEMETRY_SEEN.clear()
    _worker._install_runtime_api()
    _worker._namespace["model"] = model
    _worker._namespace["alias"] = model
    _worker._namespace["bundle"] = {"nested": {"inner": model}}
    try:
        first = _worker._execute("model.log.append({'query_type': 'aggregate', 'returned_rows': 3})")
        assert first["ok"] and [c["query_type"] for c in first["source_calls"]] == ["aggregate"]
        assert first["source_calls"][0]["input"] == "model"
        second = _worker._execute("x = 1")
        assert "source_calls" not in second
        third = _worker._execute("model.log.append({'query_type': 'dax'}); model.log.append({'query_type': 'measure'})")
        assert [c["query_type"] for c in third["source_calls"]] == ["dax", "measure"]
    finally:
        _worker._namespace.clear()
        _worker._TELEMETRY_SEEN.clear()


def test_exec_result_and_interpreter_merge_parent_side_lakehouse_calls(monkeypatch) -> None:
    raw = {"ok": True, "submitted": False, "stdout": "", "stderr": "", "state": {}, "source_calls": [{"query_type": "dax"}, "junk"]}
    assert ExecResult.from_response(raw).source_calls == [{"query_type": "dax"}]
    assert ExecResult.from_response({"ok": True}).source_calls is None

    interpreter = Interpreter(timeout=5)
    interpreter._pending_source_calls = []
    monkeypatch.setattr(
        "fabric_rlm.interpreter._execute_bound_lakehouse_query",
        lambda sources, kwargs: {"columns": ["a"], "rows": [[1], [2]], "truncated": False},
    )
    result = interpreter._timed_lakehouse_query({"root": "abfss://x/Tables", "sql": "SELECT 1", "sources": {"t": "t"}, "max_rows": 10})
    assert result["rows"] == [[1], [2]]
    (record,) = interpreter._pending_source_calls
    assert record["query_type"] == "lakehouse_sql" and record["returned_rows"] == 2
    assert record["source_root"] == "abfss://x/Tables" and "SELECT" not in str(record)

    def boom(sources, kwargs):
        raise ValueError("bad sql")

    monkeypatch.setattr("fabric_rlm.interpreter._execute_bound_lakehouse_query", boom)
    with pytest.raises(ValueError):
        interpreter._timed_lakehouse_query({"root": "abfss://x/Tables", "sql": "SELECT"})
    assert interpreter._pending_source_calls[-1]["reason"] == "execution_error"


# -- harvesting ---------------------------------------------------------------


def test_harvest_turns_typed_telemetry_into_evidence_without_data() -> None:
    result = _result(
        [
            _turn(1, "m.aggregate(measures=['ARR $'], groupby=WIDE)", calls=[{"input": "arr_model", **AGGREGATE_TOO_BROAD}]),
            _turn(2, "coarse = arr_model.aggregate(['ARR $', 'ARR $ Previous Period'], groupby=['Products[Product]', 'Sold To[Region]'])\nbad = arr_model.aggregate(['ARR Growth'])", calls=[{"input": "arr_model", **AGGREGATE_OK}, {"input": "arr_model", **AGGREGATE_UNKNOWN}]),
            _turn(3, "worst = coarse.sort_values('arr').head(10)\ncandidates = restrict_to_candidate_tuples(hist, worst, keys=['product', 'region'])\nrows = lh.query('select 1')", calls=[LAKEHOUSE_OK]),
            _turn(4, "keys = candidates[['product', 'region']].drop_duplicates()\nfine = arr_model.aggregate(['ARR $'], groupby=['Products[Product]', 'Sold To[Customer Group]', 'Sold To[Region]'], filters={'Products[Product]': list(keys['product'])})", calls=[{"input": "arr_model", **AGGREGATE_OK, "filter_count": 1, "filter_columns": ["Period[YearQuarter]"], "groupby": ["Products[Product]", "Sold To[Customer Group]", "Sold To[Region]"], "measure_identities": []}]),
            _turn(5, "SUBMIT(answer='x')", submitted=True),
        ],
        metadata={"analytical_integrity_mode": "repair", "verifier_execution": {"verified": True, "passed": ["output_validator"]}},
    )

    class Lakehouse:
        root = "abfss://ws@onelake/lh/Tables"

    evidence = harvest_evidence(result, sources={"arr_model": object(), "lake": Lakehouse()})
    by_type = {}
    for record in evidence:
        by_type.setdefault(record.observation_type, []).append(record)
    queries = by_type["query_execution"]
    assert [r.execution_status for r in queries] == ["rejected", "success", "rejected", "success", "success"]
    assert {r.source_ids for r in queries} == {("arr_model",), ("lake",)}
    ok = queries[1]
    assert ok.observation["grain"] == ("Products[Product]", "Sold To[Region]")
    assert ok.observation["measure_identities"] == (("ARR $", "ARR $ Previous Period"),)
    assert "query" not in ok.observation and "input" not in ok.observation
    assert ok.verifier_status == "passed" and ok.analytical_integrity_status == "passed"
    unknown = queries[2]
    assert unknown.observation["invalid_reference"] == "ARR Growth"
    assert unknown.observation["invalid_reference_kind"] == "measure"
    assert unknown.observation["error_class"] == "SemanticModelQueryError"
    assert "error" not in unknown.observation
    lake = queries[3]
    assert lake.source_ids == ("lake",) and "source_root" not in lake.observation

    (sequence,) = by_type["strategy_sequence"]
    assert sequence.source_ids == ("arr_model",)
    assert [dict(step)["filtered"] for step in sequence.observation["steps"]] == [False, True]
    # the restriction call between the coarse and the fine query is what proves the strategy
    assert sequence.observation["strategy"] == "candidate_drilldown"
    assert sequence.observation["candidate_identity_preserved"] is True
    assert sequence.observation["candidate_source_grain"] == ("Products[Product]", "Sold To[Region]")
    assert sequence.analytically_trusted
    (outcome,) = by_type["run_outcome"]
    assert outcome.observation["source_calls"] == 5
    assert outcome.observation["failed_source_calls"] == 2
    assert outcome.observation["first_useful_query_turn"] == 2
    assert outcome.source_ids == ("arr_model", "lake")
    assert all(r.run_fingerprint == run_fingerprint_for(result) for r in evidence)
    assert len({r.evidence_id for r in evidence}) == len(evidence)
    # every record survives the package contract and persistence validator
    for record in evidence:
        EvidenceRecord.from_dict(record.to_dict())

    summary = source_call_summary(result.trajectory.turns)
    assert summary == {"source_calls": 5, "failed_source_calls": 2, "source_seconds": pytest.approx(16.4), "first_useful_query_turn": 2}


def test_harvest_reflects_run_outcome_and_restricts_to_known_sources() -> None:
    failed = _result(
        [_turn(1, "m.aggregate(...)", calls=[{"input": "arr_model", **AGGREGATE_OK}], error="ValueError: boom")],
        submitted=False,
        failure_reason="max_turns",
        metadata={"analytical_integrity_mode": "repair", "analytical_integrity_unresolved": ["x"], "verifier_execution": {"verified": True, "passed": ["output_validator"]}},
    )
    (query, outcome) = harvest_evidence(failed, sources={"arr_model": object()})
    assert query.verifier_status == "failed" and query.analytical_integrity_status == "unresolved"
    assert query.execution_trusted and not query.analytically_trusted
    assert outcome.observation["failure_reason"] == "max_turns" and outcome.observation["error_classes"] == ("ValueError",)

    # a submission nobody verified is "none", never "passed"
    unchecked = _result([_turn(1, "m.aggregate(...)", calls=[{"input": "arr_model", **AGGREGATE_OK}])], metadata={"analytical_integrity_mode": "repair"})
    (query, _outcome) = harvest_evidence(unchecked, sources={"arr_model": object()})
    assert query.verifier_status == "none" and not query.analytically_trusted

    # a finer filtered query without a candidate restriction is only a sequence,
    # and so is one with a restriction of something unrelated in between
    fine_call = {"input": "arr_model", **AGGREGATE_OK, "filter_count": 1, "groupby": ["Products[Product]", "Sold To[Customer Group]", "Sold To[Region]"]}
    for turns in (
        [
            _turn(1, "coarse = arr_model.aggregate(['ARR $'], groupby=['Products[Product]', 'Sold To[Region]'])", calls=[{"input": "arr_model", **AGGREGATE_OK}]),
            _turn(2, "fine = arr_model.aggregate(['ARR $'], groupby=[...], filters={'Period[YearQuarter]': ['2026/Q2']})", calls=[fine_call]),
        ],
        [
            _turn(1, "coarse = arr_model.aggregate(['ARR $'], groupby=['Products[Product]', 'Sold To[Region]'])", calls=[{"input": "arr_model", **AGGREGATE_OK}]),
            _turn(2, "other = restrict_to_candidate_tuples(lookup, watchlist, keys=['product'])", calls=[]),
            _turn(3, "fine = arr_model.aggregate(['ARR $'], groupby=[...], filters={'Products[Product]': list(other['product'])})", calls=[fine_call]),
        ],
        [
            _turn(1, "coarse = arr_model.aggregate(['ARR $'], groupby=['Products[Product]', 'Sold To[Region]'])", calls=[{"input": "arr_model", **AGGREGATE_OK}]),
            _turn(2, "candidates = restrict_to_candidate_tuples(hist, coarse, keys=['product', 'region'])", calls=[]),
            _turn(3, "fine = arr_model.aggregate(['ARR $'], groupby=[...], filters={'Period[YearQuarter]': ['2026/Q2']})", calls=[fine_call]),
        ],
    ):
        plain = _result(turns, metadata={"analytical_integrity_mode": "repair", "verifier_execution": {"verified": True, "passed": ["output_validator"]}})
        sequence = next(r for r in harvest_evidence(plain, sources={"arr_model": object()}) if r.observation_type == "strategy_sequence")
        assert "strategy" not in sequence.observation and "candidate_identity_preserved" not in sequence.observation

    off = _result([_turn(1, "x=1", calls=[{"input": "other", **AGGREGATE_OK}])], metadata={"analytical_integrity_mode": "off"})
    records = harvest_evidence(off, sources={"arr_model": object()}, known_source_ids=["arr_model"])
    assert [r.observation_type for r in records] == ["run_outcome"]
    assert records[0].analytical_integrity_status == "off"
    mapped = harvest_evidence(off, known_source_ids=["arr_model"], aliases={"other": "arr_model"}, source_fingerprints={"arr_model": "schema-1"})
    assert mapped[0].source_ids == ("arr_model",) and mapped[0].source_fingerprints == {"arr_model": "schema-1"}


# -- capture is observational ---------------------------------------------------


def test_capture_on_and_off_produce_the_same_prompts_and_answer(tmp_path: Path) -> None:
    source = _csv(tmp_path)
    script = _code("SUBMIT(answer=orders.name)")
    runs = {}
    for capture in (False, True):
        lm = ScriptedLM(script)
        result = RLM.task(
            "Return the approved source file name.",
            inputs={"orders": source},
            outputs=["answer"],
            lm=lm,
            max_turns=1,
            timeout=10,
            capture_evidence=capture,
        ).run()
        runs[capture] = (result, lm.messages)
    off, on = runs[False], runs[True]
    assert off[0].payload == on[0].payload == {"answer": "orders.csv"}
    assert off[1] == on[1]
    assert [t.code for t in off[0].turns] == [t.code for t in on[0].turns]
    assert off[0].evidence == ()
    assert [r.observation_type for r in on[0].evidence] == ["run_outcome"]
    assert on[0].evidence[0].source_ids == ("orders",)
    # no validator or skill verifier was configured, so the run is unverified
    assert on[0].evidence[0].verifier_status == "none"
    assert on[0].trajectory.metadata["verifier_configured"] is False

    lm = ScriptedLM(script)
    checked = RLM.task(
        "Return the approved source file name.",
        inputs={"orders": source},
        outputs=["answer"],
        lm=lm,
        max_turns=1,
        timeout=10,
        capture_evidence=True,
        output_validator=lambda payload: None,
    ).run()
    assert checked.trajectory.metadata["verifier_configured"] is True
    assert checked.evidence[0].verifier_status == "passed"


# -- guidance injection ---------------------------------------------------------


def _knowledge_with_lesson(source: Path, *, status: str) -> Knowledge:
    learned = RLM.learn(sources={"orders": source})
    profile = learned.package.sources[0]
    lesson = LearnedLesson(
        lesson_id="lesson.valid_grain.orders",
        kind="valid_grain",
        subject="region",
        structured_rule={"grain": ["region"], "measures": ["amount"], "max_rows_observed": 4, "advice": "reliable_analysis_grain"},
        status=status,
        confidence="medium",
        source_dependencies=("orders",),
        source_fingerprints={"orders": profile.schema_fingerprint},
        basis=("verified_success",),
    )
    package = replace(learned.package, lessons=(lesson,))
    return Knowledge(package=package, bindings=learned.bindings, _registry=learned._registry, _limits=learned._limits)


def test_active_lessons_are_injected_after_the_inputs_and_the_source_stays_bound(tmp_path: Path) -> None:
    source = _csv(tmp_path)
    knowledge = _knowledge_with_lesson(source, status="active")
    lm = ScriptedLM(_code("SUBMIT(answer=orders.name)"))
    result = RLM.task(
        "Total amount by region",
        outputs=["answer"],
        knowledge=knowledge,
        lm=lm,
        max_turns=1,
        timeout=10,
    ).run()
    prompt = _system_prompt(lm)
    assert result.payload == {"answer": "orders.csv"}
    inputs_at = prompt.index("## Inputs available in namespace")
    guidance_at = prompt.index("## Learned source guidance")
    outputs_at = prompt.index("## Required output fields for SUBMIT()")
    assert inputs_at < guidance_at < outputs_at
    assert "- [orders] amount by region executed successfully in prior runs (up to 4 rows): an observed feasible query grain" in prompt
    assert "  orders:" in prompt
    assert result.trajectory.metadata["knowledge_lessons_injected"] == ["lesson.valid_grain.orders"]
    assert result.trajectory.metadata["knowledge_lessons_available"] == 1


def test_lessons_reach_the_operation_planner_and_the_synthesis_prompt(tmp_path: Path) -> None:
    """The planner sees the guidance so it can add the context a measure needs.

    When the host then executes the operation, the agent that interprets
    the packet keeps the guidance too: a grouped result still has a
    "current" row to pick.
    """
    source = _csv(tmp_path)
    knowledge = _knowledge_with_lesson(source, status="active")
    lm = ScriptedLM('{"fallback": true, "reason": "test"}', _code("SUBMIT(answer=orders.name)"))
    result = RLM.task("Total amount by region", outputs=["answer"], knowledge=knowledge, lm=lm, max_turns=1, timeout=10).run()
    planner = lm.messages[0][1]["content"]
    assert "Registered operations:" in planner
    assert '"required_sources":["orders"]' in planner
    assert "## Learned source guidance" in planner and "observed feasible query grain" in planner
    assert "Apply this guidance" in planner
    assert result.trajectory.metadata["knowledge_mode"] == "fallback_no_compatible_operation"
    assert "## Learned source guidance" in _system_prompt(lm)

    plan = '{"operation_id": "orders.tabular.aggregate.v1", "parameters": {"aggregate": "sum", "measure": "amount"}}'
    lm = ScriptedLM(plan, _code("SUBMIT(answer=knowledge_result['row_count'])"))
    result = RLM.task("Total amount by region", outputs=["answer"], knowledge=knowledge, lm=lm, max_turns=1, timeout=10).run()
    assert result.trajectory.metadata["knowledge_mode"] == "registered_operation"
    assert "## Learned source guidance" in lm.messages[0][1]["content"]
    assert "## Learned source guidance" in _system_prompt(lm)
    assert "knowledge_result" in _system_prompt(lm)
    assert result.trajectory.metadata["knowledge_lessons_injected"] == ["lesson.valid_grain.orders"]


def test_planner_guidance_is_scoped_to_the_sources_its_operations_read(tmp_path: Path) -> None:
    """A lesson about a source no operation reads never reaches the planner."""
    source = _csv(tmp_path)
    other = tmp_path / "notes.txt"
    other.write_text("free text", encoding="utf-8")
    learned = RLM.learn(sources={"orders": source, "notes": other}, roles={"notes": "context_only"})
    assert {op.required_sources for op in learned.package.operations} == {("orders",)}
    profile = next(p for p in learned.package.sources if p.source_id == "notes")
    lesson = LearnedLesson(
        lesson_id="lesson.semantic_fact.notes",
        kind="semantic_fact",
        subject="amount",
        structured_rule={"fact": "amount_is_net_of_tax", "measures": ["amount"]},
        status="active",
        confidence="medium",
        source_dependencies=("notes",),
        source_fingerprints={"notes": profile.schema_fingerprint},
    )
    knowledge = Knowledge(package=replace(learned.package, lessons=(lesson,)), bindings=learned.bindings, _registry=learned._registry, _limits=learned._limits)
    lm = ScriptedLM('{"fallback": true, "reason": "test"}', _code("SUBMIT(answer=orders.name)"))
    RLM.task("Total amount by region", outputs=["answer"], knowledge=knowledge, lm=lm, max_turns=1, timeout=10).run()
    planner = lm.messages[0][1]["content"]
    assert "Learned source guidance" not in planner
    agent = _system_prompt(lm)
    assert "- [notes] amount: fact amount_is_net_of_tax" in agent


def test_candidates_and_unrelated_lessons_leave_the_prompt_untouched(tmp_path: Path) -> None:
    source = _csv(tmp_path)
    prompts = {}
    for label, knowledge in (
        ("candidate", _knowledge_with_lesson(source, status="candidate")),
        ("unrelated", _knowledge_with_lesson(source, status="active")),
        ("none", RLM.learn(sources={"orders": source})),
    ):
        lm = ScriptedLM(_code("SUBMIT(answer=orders.name)"))
        RLM.task("How many rows does the file have?", outputs=["answer"], knowledge=knowledge, lm=lm, max_turns=1, timeout=10).run()
        prompts[label] = _system_prompt(lm)
    assert prompts["candidate"] == prompts["none"] == prompts["unrelated"]
    assert "Learned source guidance" not in prompts["none"]
    assert build_system_prompt(inline_task="t", inputs={}, learned_guidance=None) == build_system_prompt(inline_task="t", inputs={})


# -- enrich round trip -----------------------------------------------------------


def test_enrich_learns_from_results_and_round_trips_through_the_store(tmp_path: Path) -> None:
    source = _csv(tmp_path)
    store = tmp_path / "orders.knowledge.json"
    knowledge = RLM.learn(sources={"orders": source}, store=store)
    assert knowledge.package.format_version == 1

    lm = ScriptedLM(_code("SUBMIT(answer=orders.name)"))
    result = RLM.task("Return the file name", outputs=["answer"], knowledge=knowledge, lm=lm, max_turns=1, timeout=10, capture_evidence=True).run()
    assert [r.observation_type for r in result.evidence] == ["run_outcome"]

    enriched = RLM.enrich(knowledge, [result], store=store, overwrite=True)
    assert knowledge.package.evidence == ()
    assert enriched.package.format_version == 2
    assert [r.observation_type for r in enriched.package.evidence] == ["run_outcome"]
    assert enriched.package.evidence[0].source_fingerprints == {"orders": knowledge.package.sources[0].schema_fingerprint}
    assert enriched.bindings == knowledge.bindings

    loaded = load_knowledge(store, sources={"orders": Path(str(source))})
    assert loaded.package == enriched.package
    again = RLM.enrich(loaded, [result])
    assert len(again.package.evidence) == 1
    with pytest.raises(TypeError):
        RLM.enrich(loaded, ["not a result"])


# -- benchmark metrics -----------------------------------------------------------


def test_benchmark_records_source_calls_repairs_integrity_and_parity() -> None:
    task = KnowledgeBenchmarkTask(
        task_id="arr",
        question="What was total ARR this quarter?",
        expected_operation_id="arr_model.semantic_model.measure.v1",
        is_correct=lambda payload: payload == {"answer": "x"},
    )

    def run(_task, arm, _lm):
        if arm == "cold":
            return _result([
                _turn(1, "m.aggregate(...)", calls=[{"input": "arr_model", **AGGREGATE_TOO_BROAD}]),
                _turn(2, "m.aggregate(...)", calls=[{"input": "arr_model", **AGGREGATE_OK}], turn_type="verifier_repair"),
                _turn(3, "SUBMIT(answer='x')", submitted=True),
            ], metadata={"analytical_integrity_unresolved": ["noise"]})
        return _result([
            _turn(1, "m.aggregate(...)", calls=[{"input": "arr_model", **AGGREGATE_OK}]),
            _turn(2, "SUBMIT(answer='x')", submitted=True),
        ], metadata={"knowledge_lessons_injected": ["lesson.a", "lesson.b"], "knowledge_mode": "fallback_no_registered_operations"})

    class LM:
        cache = False
        kwargs: dict = {}

    report = run_knowledge_benchmark(tasks=[task], repetitions=1, seed=1, make_lm=lambda **_k: LM(), run=run)
    rows = {trial.arm: trial for trial in report.trials}
    assert rows["cold"].source_calls == 2 and rows["cold"].failed_source_calls == 1
    assert rows["cold"].first_useful_query_turn == 2 and rows["cold"].verifier_repairs == 1
    assert rows["cold"].integrity_ok is False and rows["learned"].integrity_ok is True
    assert rows["learned"].lessons_injected == 2 and rows["learned"].source_seconds == pytest.approx(3.5)
    parity = report.cold_parity()
    assert parity["parity"] is True and parity["turns_delta"] == -1.0 and parity["source_calls_delta"] == -1.0
    assert parity["task_results"] == {"arr": {"cold": 1.0, "learned": 1.0, "ok": True}}
    summary = report.summary()
    assert summary["cold"]["mean_failed_source_calls"] == 1.0 and summary["learned"]["integrity_ok_rate"] == 1.0


def test_cold_parity_fails_when_any_task_regresses_even_if_the_average_improves() -> None:
    tasks = [
        KnowledgeBenchmarkTask("lookup", "What is total ARR this quarter?", "op", lambda p: p == {"answer": "x"}),
        KnowledgeBenchmarkTask("decline", "Which segments lost the most ARR?", "op", lambda p: p == {"answer": "x"}),
    ]
    outcomes = {("lookup", "cold"): False, ("lookup", "learned"): True, ("decline", "cold"): True, ("decline", "learned"): False}

    def run(task, arm, _lm):
        answer = "x" if outcomes[(task.task_id, arm)] else "wrong"
        return _result([_turn(1, f"SUBMIT(answer='{answer}')", submitted=True)], payload={"answer": answer})

    class LM:
        cache = False
        kwargs: dict = {}

    # both arms score 50% overall; the learned arm regressed on one task
    report = run_knowledge_benchmark(tasks=tasks, repetitions=1, seed=3, make_lm=lambda **_k: LM(), run=run)
    parity = report.cold_parity()
    assert parity["correctness_ok"] is True
    assert parity["task_results"]["decline"] == {"cold": 1.0, "learned": 0.0, "ok": False}
    assert parity["per_task_correctness_ok"] is False and parity["parity"] is False

    # a task missing one arm is not proven either
    from fabric_rlm.knowledge_benchmark import KnowledgeBenchmarkReport

    partial = KnowledgeBenchmarkReport(seed=1, repetitions=1, trials=tuple(t for t in report.trials if t.arm == "cold"))
    assert partial.cold_parity()["task_results"]["lookup"]["ok"] is False
    assert partial.cold_parity()["parity"] is False


# -- verification is execution, not configuration -----------------------------


def _skill_loader(tmp_path: Path, name: str, verifier: str | None):
    import textwrap

    from fabric_rlm.skill_loader import SkillLoader

    body = textwrap.dedent(
        f"""\
        ---
        applies_when:
          keywords: [{name}]
        specificity: domain
        ---

        # {name} skill

        Body.
        """
    )
    if verifier is not None:
        body += "\n## Required verifier\n\n```python\n" + verifier + "\n```\n"
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{name}.md").write_text(body, encoding="utf-8")
    return SkillLoader(skill_dir=tmp_path)


def _checked_run(tmp_path: Path, **kwargs):
    source = _csv(tmp_path)
    return RLM.task(
        "Return the approved source file name.",
        inputs={"orders": source},
        outputs=["answer"],
        lm=ScriptedLM(_code("SUBMIT(answer=orders.name)")),
        max_turns=1,
        timeout=60,
        capture_evidence=True,
        **kwargs,
    ).run()


def test_verifier_status_records_what_ran_not_what_was_configured(tmp_path: Path) -> None:
    # a validator that executed and accepted the answer
    passed = _checked_run(tmp_path, output_validator=lambda payload: None)
    execution = passed.trajectory.metadata["verifier_execution"]
    assert execution["passed"] == ["output_validator"] and execution["verified"] is True
    assert passed.evidence[0].verifier_status == "passed"

    # a validator that crashed: the runtime accepts the answer (graceful
    # degrade) but nothing verified it
    def broken(payload):
        raise RuntimeError("validator bug")

    degraded = _checked_run(tmp_path, output_validator=broken)
    assert degraded.submitted and degraded.payload == {"answer": "orders.csv"}
    execution = degraded.trajectory.metadata["verifier_execution"]
    assert execution["degraded"] == ["output_validator"] and execution["verified"] is False
    assert degraded.trajectory.metadata["verifier_configured"] is True
    assert degraded.evidence[0].verifier_status == "none"

    # a loaded skill without a verifier configures nothing
    loader = _skill_loader(tmp_path / "plain", "plain", None)
    unchecked = _checked_run(tmp_path, skills=["plain"], skill_loader=loader, enable_router=False)
    assert unchecked.submitted
    assert unchecked.trajectory.metadata["verifier_configured"] is False
    assert unchecked.trajectory.metadata["verifier_execution"]["checks"] == []
    assert unchecked.evidence[0].verifier_status == "none"

    # a skill verifier that ran and accepted the answer
    loader = _skill_loader(
        tmp_path / "strict",
        "strict",
        "def verify(payload):\n    assert payload.get('answer') == 'orders.csv'",
    )
    checked = _checked_run(tmp_path, skills=["strict"], skill_loader=loader, enable_router=False)
    assert checked.submitted
    execution = checked.trajectory.metadata["verifier_execution"]
    assert execution["passed"] == ["skill:strict"] and execution["verified"] is True
    assert checked.evidence[0].verifier_status == "passed"


def test_run_ids_keep_identical_executions_apart(tmp_path: Path) -> None:
    first = _checked_run(tmp_path, output_validator=lambda payload: None)
    second = _checked_run(tmp_path, output_validator=lambda payload: None)
    assert [t.code for t in first.turns] == [t.code for t in second.turns]
    assert first.payload == second.payload
    assert first.trajectory.metadata["run_id"] != second.trajectory.metadata["run_id"]
    assert run_fingerprint_for(first) != run_fingerprint_for(second)
    # harvesting the same execution again is the same observation
    again = harvest_evidence(first, sources={"orders": object()})
    assert {r.run_fingerprint for r in again} == {r.run_fingerprint for r in first.evidence}
    assert {r.evidence_id for r in again} == {r.evidence_id for r in first.evidence}
    # a result without an id keeps the content fingerprint
    bare = _result([_turn(1, "x=1")])
    assert run_fingerprint_for(bare) == run_fingerprint_for(_result([_turn(1, "x=1")]))
