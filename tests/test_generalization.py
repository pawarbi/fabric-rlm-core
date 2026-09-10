"""Generalization across domains and source types, with scripted models.

Every model turn is scripted, so each test is about the library: the
execution protocol, the abstention path, claims tracing, the registered
operation packet contract, and what a file or Lakehouse source can learn.
Three synthetic domains stand in for the world: a manufacturing production
log (CSV), an inventory snapshot (Parquet) and a service ticket table
(Delta, through a LakehouseSource).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fabric_rlm import RLM
from fabric_rlm.knowledge import KnowledgePackage
from fabric_rlm.knowledge_evidence import harvest_evidence
from fabric_rlm.knowledge_lessons import declared_lessons
from fabric_rlm.knowledge_retrieval import render_learned_guidance, retrieve_lessons


class ScriptedLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict[str, str]]] = []

    def __call__(self, *, messages):
        self.messages.append([dict(message) for message in messages])
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def _code(body: str) -> str:
    return f"```python\n{body}\n```"


def _last_user(lm: ScriptedLM) -> str:
    return next(m["content"] for m in reversed(lm.messages[-1]) if m["role"] == "user")


def _system(lm: ScriptedLM) -> str:
    return lm.messages[-1][0]["content"]


TASK = "Total produced units across lines where reporting is complete."
COMPLETE_TOTAL = 1200 + 800 + 1300 + 700 + 600


def _production_csv(tmp_path: Path) -> Path:
    path = tmp_path / "production.csv"
    path.write_text(
        "line,period,produced_units,defect_units,reporting_complete,is_current_period\n"
        "L1,2024-01,1200,30,true,false\n"
        "L2,2024-01,800,20,true,false\n"
        "L3,2024-01,500,40,false,false\n"
        "L1,2024-02,1300,25,true,true\n"
        "L2,2024-02,700,15,true,true\n"
        "L3,2024-02,600,35,true,true\n",
        encoding="utf-8",
    )
    return path


def _inventory_parquet(tmp_path: Path) -> Path:
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    path = tmp_path / "inventory.parquet"
    pandas.DataFrame(
        {
            "warehouse": ["W1", "W1", "W2", "W2"],
            "product": ["A", "B", "A", "B"],
            "snapshot_date": ["2026-03-31"] * 4,
            "on_hand": [50, 20, 70, 46],
        }
    ).to_parquet(path, index=False)
    return path


def _service_lakehouse(tmp_path: Path):
    deltalake = pytest.importorskip("deltalake")
    pyarrow = pytest.importorskip("pyarrow")
    from fabric_rlm.lakehouse import LakehouseSource

    tickets = tmp_path / "tickets"
    deltalake.write_deltalake(
        str(tickets),
        pyarrow.table(
            {
                "ticket_id": [1, 2, 3, 4],
                "region": ["EU", "EU", "US", "US"],
                "resolved": [True, False, True, True],
                "hours": [2.0, 5.0, 1.0, 3.0],
            }
        ),
    )
    table = deltalake.DeltaTable(str(tickets), without_files=True)
    return LakehouseSource(
        "file:///service-lakehouse",
        catalog=[
            {
                "kind": "delta",
                "name": "tickets",
                "path": str(tickets),
                "version": table.version(),
                "table_id": table.metadata().id,
                "columns": [
                    ["ticket_id", "BIGINT"],
                    ["region", "VARCHAR"],
                    ["resolved", "BOOLEAN"],
                    ["hours", "DOUBLE"],
                ],
            }
        ],
    )


# ------------------------------------------------------------ protocol --


def test_dependent_code_blocks_run_in_order_and_fabricated_output_is_named(tmp_path: Path) -> None:
    # The transcript shape that broke the manufacturing run: three blocks,
    # invented output between them, a last block that needs the first.
    transcript = (
        "Load the data.\n\n"
        + _code("import pandas as pd\ndf = pd.read_csv(production)\ncomplete = df[df['reporting_complete']]")
        + "\n```\n(6 rows loaded)\n```\n\nNow the total.\n\n"
        + _code("total = int(complete['produced_units'].sum())\nprint('total', total)")
        + "\n```\ntotal 72800\n```\n"
    )
    lm = ScriptedLM(transcript, _code("SUBMIT(answer={'value': total})"))
    result = RLM.task(TASK, inputs={"production": _production_csv(tmp_path)}, outputs=["answer"], lm=lm, max_turns=3, timeout=60).run()

    first = result.turns[0]
    assert first.error is None
    assert "df = pd.read_csv(production)" in first.code and "SUBMIT" not in first.code
    assert first.stdout.strip() == f"total {COMPLETE_TOTAL}"
    feedback = lm.messages[1][-1]["content"]
    assert "Protocol note" in feedback and "executed in order" in feedback
    assert "not executed and is not evidence" in feedback
    assert result.submitted and result.payload["answer"]["value"] == COMPLETE_TOTAL


def test_a_revision_still_runs_alone(tmp_path: Path) -> None:
    revision = "Sketch:\n" + _code("x = broken(") + "\nCorrected:\n" + _code("x = 2 + 2\nSUBMIT(answer=x)")
    lm = ScriptedLM(revision)
    result = RLM.task("Add.", outputs=["answer"], lm=lm, max_turns=2, timeout=60).run()
    assert result.submitted and result.payload["answer"] == 4
    assert result.turns[0].code == "x = 2 + 2\nSUBMIT(answer=x)"


def test_abstain_ends_the_run_on_the_record(tmp_path: Path) -> None:
    lm = ScriptedLM(_code("ABSTAIN('no row has reporting_complete for 2024-03')"))
    result = RLM.task(TASK, inputs={"production": _production_csv(tmp_path)}, outputs=["answer"], lm=lm, max_turns=3, timeout=60).run()
    assert result.submitted is False and result.payload is None
    assert result.failure_reason == "abstained"
    assert result.trajectory.metadata["abstain_reason"] == "no row has reporting_complete for 2024-03"
    assert len(result.turns) == 1

    # the conventional import form works too
    lm = ScriptedLM(_code("from sandbox import ABSTAIN\nABSTAIN('cannot compute')"))
    result = RLM.task("Anything.", outputs=["answer"], lm=lm, max_turns=2, timeout=60).run()
    assert result.failure_reason == "abstained"


def test_final_turn_asks_for_computed_values_or_an_abstention() -> None:
    lm = ScriptedLM(_code("raise RuntimeError('boom')"), _code("ABSTAIN('nothing executed')"))
    result = RLM.task("Anything.", outputs=["answer"], lm=lm, max_turns=2, timeout=60).run()
    final = lm.messages[1][-1]["content"]
    assert "FINAL TURN" in final
    assert "ABSTAIN" in final and "computed by code that actually ran" in final
    assert "strictly better" not in final
    assert result.failure_reason == "abstained"
    assert "call ABSTAIN" in _system(lm)


def test_numbers_typed_into_submit_must_appear_in_executed_output(tmp_path: Path) -> None:
    # value 0 and "72,800 units" were never printed by anything that ran
    lm = ScriptedLM(
        _code("import pandas as pd\ndf = pd.read_csv(production)\ntotal = int(df[df['reporting_complete']]['produced_units'].sum())"),
        _code("SUBMIT(answer={'value': 0, 'note': 'about 72,800 units were produced'})"),
        _code("print('total', total)\nSUBMIT(answer={'value': total, 'note': 'computed from complete rows'})"),
    )
    result = RLM.task(TASK, inputs={"production": _production_csv(tmp_path)}, outputs=["answer"], lm=lm, max_turns=4, timeout=60).run()
    rejection = lm.messages[2][-1]["content"]
    assert "analytical integrity" in rejection
    assert "The value 0 was typed into SUBMIT" in rejection
    assert "The value 72,800 was typed into SUBMIT" in rejection
    assert result.submitted and result.payload["answer"]["value"] == COMPLETE_TOTAL
    assert result.integrity_ok
    history = result.trajectory.metadata["verifier_repair_history"]
    assert history[0]["skill"] == "analytical_integrity"


def test_a_typed_number_the_output_showed_is_accepted(tmp_path: Path) -> None:
    lm = ScriptedLM(
        _code("import pandas as pd\ndf = pd.read_csv(production)\nprint('rows', len(df))"),
        _code("SUBMIT(answer={'rows': 6, 'period': '2024'})"),
    )
    result = RLM.task("How many rows are there for 2024?", inputs={"production": _production_csv(tmp_path)}, outputs=["answer"], lm=lm, max_turns=3, timeout=60).run()
    assert result.submitted and result.integrity_ok
    assert "verifier_repair_history" not in result.trajectory.metadata


# ------------------------------------------------ registered operations --


def _plan(operation_id: str, **parameters: str) -> str:
    return json.dumps({"operation_id": operation_id, "parameters": parameters})


def test_the_packet_introduces_itself_and_the_raw_source_stays_bound(tmp_path: Path) -> None:
    knowledge = RLM.learn(sources={"production": _production_csv(tmp_path)})
    operation = knowledge.package.operations[0]
    plan = _plan(operation.operation_id, aggregate="sum", measure="produced_units", filter_column="reporting_complete", filter_value="true")
    agent = _code(
        "import pandas as pd\n"
        "packet_total = knowledge_result['rows'][0]['value']\n"
        "raw_total = int(pd.read_csv(production).query('reporting_complete')['produced_units'].sum())\n"
        "print('packet', packet_total, 'raw', raw_total)\n"
        "SUBMIT(answer={'value': packet_total, 'cross_check': raw_total})"
    )
    lm = ScriptedLM(plan, agent)
    result = RLM.task(TASK, outputs=["answer"], knowledge=knowledge, lm=lm, max_turns=2, timeout=60).run()

    assert result.trajectory.metadata["knowledge_mode"] == "registered_operation"
    assert result.submitted and result.payload["answer"] == {"value": COMPLETE_TOTAL, "cross_check": COMPLETE_TOTAL}
    prompt = _system(lm)
    assert f"host-computed result of registered operation {operation.operation_id}" in prompt
    assert "rows: 1 row(s) with columns ['value']" in prompt
    assert '"filter_column": "reporting_complete"' in prompt
    assert "already aggregated result, not the raw table" in prompt
    assert "Raw source(s) production remain bound" in prompt
    assert "production:" in prompt  # the raw handle is still listed as an input


def test_host_operations_are_evidence_and_a_file_source_learns_its_grain(tmp_path: Path) -> None:
    knowledge = RLM.learn(sources={"production": _production_csv(tmp_path)})
    operation = knowledge.package.operations[0]
    plan = _plan(operation.operation_id, aggregate="sum", measure="produced_units", groupby="line")
    agent = _code("print(knowledge_result['rows'])\nSUBMIT(answer={'rows': knowledge_result['rows']})")
    runs = []
    for _ in range(2):
        lm = ScriptedLM(plan, agent)
        runs.append(
            RLM.task("Produced units by line.", outputs=["answer"], knowledge=knowledge, lm=lm, max_turns=2, timeout=60, capture_evidence=True).run()
        )
    execution = runs[0].trajectory.metadata["operation_execution"]
    assert execution["query_type"] == "registered_operation" and execution["groupby"] == ["line"]
    assert execution["returned_rows"] == 3 and execution["executed"] is True
    queries = [e for e in runs[0].evidence if e.observation_type == "query_execution"]
    assert [q.observation["query_type"] for q in queries] == ["registered_operation"]
    assert queries[0].source_ids == ("production",) and tuple(queries[0].observation["grain"]) == ("line",)
    assert queries[0].execution_status == "success"

    # two separate runs at the same grain: an active valid-grain lesson
    enriched = RLM.enrich(knowledge, runs)
    grain_lessons = [l for l in enriched.package.lessons if l.kind == "valid_grain"]
    assert [(l.status, tuple(l.structured_rule["grain"]), l.structured_rule["runs"]) for l in grain_lessons] == [("active", ("line",), 2)]
    lessons = retrieve_lessons(enriched.package, "Produced units by line for complete reports")
    assert any(l.kind == "valid_grain" for l in lessons)
    assert "by line executed successfully in prior runs" in render_learned_guidance(lessons)

    # a plan the contract rejects is evidence too, without a row count
    lm = ScriptedLM(_plan(operation.operation_id, aggregate="sum", measure="produced_units", filter_column="produced_units", filter_value="lots"), _code("SUBMIT(answer=1)"))
    rejected = RLM.task("Anything.", outputs=["answer"], knowledge=knowledge, lm=lm, max_turns=2, timeout=60, capture_evidence=True).run()
    execution = rejected.trajectory.metadata["operation_execution"]
    assert execution["executed"] is False and execution["reason"] == "plan_rejected"
    assert [e.execution_status for e in rejected.evidence if e.observation_type == "query_execution"] == ["rejected"]


# ---------------------------------------------------- declared metadata --


def test_declared_facts_reach_every_task_and_are_validated(tmp_path: Path) -> None:
    declared = {
        "production": {
            "grain": ["line", "period"],
            "period_column": "period",
            "units": {"produced_units": "units", "defect_units": "units"},
            "definitions": {"reporting_complete": "the row is final; incomplete rows must be excluded from totals"},
            "notes": ["defect_units are included in produced_units"],
        }
    }
    knowledge = RLM.learn(sources={"production": _production_csv(tmp_path)}, declared=declared)
    facts = [l for l in knowledge.package.lessons if l.kind == "semantic_fact"]
    assert len(facts) == 6
    assert all(l.status == "active" and l.confidence == "high" and l.basis == ("declared",) for l in facts)
    assert all(l.source_fingerprints == {"production": knowledge.package.sources[0].schema_fingerprint} for l in facts)

    lm = ScriptedLM(_code("SUBMIT(answer='x')"))
    RLM.task("How many defect units were there?", outputs=["answer"], knowledge=knowledge, lm=lm, max_turns=1, timeout=60).run()
    prompt = _system(lm)
    assert "## Learned source guidance" in prompt
    assert "Declared grain: one row per line x period" in prompt
    assert "Declared period column: period" in prompt
    assert "defect_units is measured in units" in prompt
    assert "reporting_complete: the row is final" in prompt
    assert "defect_units are included in produced_units" in prompt
    assert "declared by the source owner" in prompt

    # the package round-trips with the declared lessons intact
    restored = KnowledgePackage.from_dict(knowledge.package.to_dict())
    assert {l.lesson_id for l in restored.lessons} == {l.lesson_id for l in knowledge.package.lessons}

    with pytest.raises(ValueError, match="unknown column: shift"):
        RLM.learn(sources={"production": _production_csv(tmp_path)}, declared={"production": {"grain": ["shift"]}})
    with pytest.raises(ValueError, match="unknown key: owner"):
        RLM.learn(sources={"production": _production_csv(tmp_path)}, declared={"production": {"owner": "ops"}})
    with pytest.raises(ValueError, match="unknown source: warehouse"):
        RLM.learn(sources={"production": _production_csv(tmp_path)}, declared={"warehouse": {"grain": ["line"]}})


def test_declared_facts_on_parquet_and_the_planner_sees_them(tmp_path: Path) -> None:
    knowledge = RLM.learn(
        sources={"inventory": _inventory_parquet(tmp_path)},
        declared={"inventory": {"grain": ["warehouse", "product"], "period_column": "snapshot_date", "units": {"on_hand": "units"}}},
    )
    operation = knowledge.package.operations[0]
    plan = _plan(operation.operation_id, aggregate="sum", measure="on_hand")
    lm = ScriptedLM(plan, _code("SUBMIT(answer={'value': knowledge_result['rows'][0]['value']})"))
    result = RLM.task("Total units on hand at the latest snapshot.", outputs=["answer"], knowledge=knowledge, lm=lm, max_turns=2, timeout=60).run()
    assert result.submitted and result.payload["answer"]["value"] == 186
    planner_prompt = lm.messages[0][-1]["content"]
    assert "Declared grain: one row per warehouse x product" in planner_prompt
    assert "Declared period column: snapshot_date" in planner_prompt


def test_a_current_period_flag_in_a_table_is_a_structural_lesson(tmp_path: Path) -> None:
    knowledge = RLM.learn(sources={"production": _production_csv(tmp_path)})
    (lesson,) = [l for l in knowledge.package.lessons if l.kind == "time_semantics"]
    assert lesson.status == "active" and lesson.confidence == "medium"
    assert tuple(lesson.structured_rule["current_period_constructs"]) == ("is_current_period",)
    assert set(lesson.basis) == {"schema_name_pattern", "boolean_period_flag"}
    lessons = retrieve_lessons(knowledge.package, "Produced units in the current period")
    assert [l.kind for l in lessons] == ["time_semantics"]
    assert retrieve_lessons(knowledge.package, "Produced units for 2024-01") == ()

    plain = tmp_path / "balances.csv"
    plain.write_text("account,current,balance\nA,true,10\n", encoding="utf-8")
    candidate = RLM.learn(sources={"balances": plain})
    (lesson,) = [l for l in candidate.package.lessons if l.kind == "time_semantics"]
    assert lesson.status == "candidate"


# ------------------------------------------------------------ lakehouse --


def test_lakehouse_queries_record_their_grain_as_evidence(tmp_path: Path) -> None:
    source = _service_lakehouse(tmp_path)
    knowledge = RLM.learn(sources={"service": source})
    lm = ScriptedLM(
        '{"fallback": true, "reason": "ad hoc"}',
        _code(
            "rows = service.query('SELECT region, COUNT(*) AS n FROM tickets WHERE resolved GROUP BY region', sources={'tickets': 'tickets'})\n"
            "print(rows['rows'])\n"
            "SUBMIT(answer={'rows': rows['rows']})"
        ),
    )
    result = RLM.task("Resolved tickets by region.", outputs=["answer"], knowledge=knowledge, lm=lm, max_turns=2, timeout=60, capture_evidence=True).run()
    assert result.submitted
    (call,) = result.turns[-1].source_calls
    assert call["query_type"] == "lakehouse_sql" and call["groupby"] == ["region"]
    assert "sql" not in call and "query" not in call
    (query,) = [e for e in result.evidence if e.observation_type == "query_execution"]
    assert query.source_ids == ("service",) and tuple(query.observation["grain"]) == ("region",)
    again = harvest_evidence(result, sources=knowledge.bindings, known_source_ids=["service"], source_fingerprints={"service": knowledge.package.sources[0].schema_fingerprint})
    assert {e.evidence_id for e in again} == {e.evidence_id for e in result.evidence}


def test_a_shared_glossary_renders_once_and_never_crowds_out_the_grain(tmp_path: Path) -> None:
    # The same definitions declared on every table of a domain, as a source
    # owner would: the agent sees the structural facts first, one line per
    # definition tagged with every table, and the evidence lessons keep their
    # own budget.
    production = _production_csv(tmp_path)
    lines = tmp_path / "lines.csv"
    lines.write_text("line,plant\nL1,North\nL2,North\nL3,South\n", encoding="utf-8")
    glossary = {
        "defect rate": "defect units divided by produced units in complete periods",
        "complete period": "only rows where reporting_complete is true",
        "units": "produced units are finished units, not started units",
    }
    knowledge = RLM.learn(
        sources={"production": production, "lines": lines},
        declared={
            "production": {"grain": ["line", "period"], "period_column": "period", "definitions": glossary},
            "lines": {"grain": ["line"], "definitions": glossary},
        },
    )
    task = "Which units had the worst defect rate in the complete periods?"
    lessons = retrieve_lessons(knowledge.package, task)
    facts = [l for l in lessons if l.kind == "semantic_fact"]
    # grain and period first, then the three definitions once each
    kinds = [l.structured_rule["fact"] for l in facts]
    assert sorted(kinds[:3]) == ["grain", "grain", "period_column"] and kinds[3:] == ["definition"] * 3
    definitions = [l for l in facts if l.structured_rule["fact"] == "definition"]
    assert all(l.source_dependencies == ("lines", "production") for l in definitions)
    assert all(set(l.source_fingerprints) == {"lines", "production"} for l in definitions)
    guidance = render_learned_guidance(lessons)
    assert guidance.count("defect units divided by produced units") == 1
    assert "[lines, production] defect rate:" in guidance
    assert "Declared grain: one row per line x period" in guidance
    # the declared budget is separate from the evidence budget and keeps the
    # structural facts first: limit=2 leaves room for four declared lines
    narrow = [l for l in retrieve_lessons(knowledge.package, task, limit=2) if l.kind == "semantic_fact"]
    narrow_kinds = [l.structured_rule["fact"] for l in narrow]
    assert sorted(narrow_kinds[:3]) == ["grain", "grain", "period_column"] and narrow_kinds[3:] == ["definition"]
