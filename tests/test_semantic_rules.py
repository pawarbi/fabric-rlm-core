"""Column rules on a SemanticModel: which column stands for a concept, checked against the queries that ran.

The case behind this: a policy-pack task on an ecommerce model where the model has a Portuguese and an English
category column and a customer state next to a postcode geography table. With the right column named in a skill,
runs still read the English column and then typed the Portuguese one into the final query. The rules judge the
latest query that touches each concept, so exploring the wrong column first is fine and the slip is not.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from fabric_rlm import RLM, ColumnRules, SemanticModel
from fabric_rlm.semantic_rules import Ref, check_calls, compile_rules

RULES = {"rules": [
    {"concept": "product category", "use": ["Products[Product Category English]"], "never": ["Products[Product Category]"]},
    {"concept": "buyer state", "use": ["Customers[Customer State]"], "never": ["Customer Geography"]},
    {"concept": "item revenue", "use": ["[Total Price]"], "never": ["Sales[Price]"], "note": "use the measure"},
]}
ENGLISH = {"query_type": "dax", "columns": ["Products[Product Category English]", "Sales[Order Status]"], "measures": ["Total Price"]}
PORTUGUESE = {"query_type": "dax", "columns": ["Products[Product Category]", "Sales[Order Status]"], "measures": ["Total Price"]}


def rules():
    return ColumnRules.from_value(RULES)


def test_refs_parse_columns_measures_and_tables():
    assert Ref.parse("'Customer Geography'[State]") == Ref("column", "Customer Geography", "State")
    assert Ref.parse("[Total Price]") == Ref("measure", "", "Total Price")
    assert Ref.parse("Customer Geography") == Ref("table", "Customer Geography", "")
    assert Ref.parse("Customer Geography").matches({"columns": ["Customer Geography[State]"]})
    assert Ref.parse("[total price]").matches({"measures": ["Total Price"]})


def test_the_latest_query_on_a_concept_decides():
    assert rules().violations([PORTUGUESE, ENGLISH]) == []          # explored the wrong column, computed with the right one
    (problem,) = rules().violations([ENGLISH, PORTUGUESE])          # read the right one, computed with the wrong one
    assert "Products[Product Category]" in problem and "Use Products[Product Category English]" in problem


def test_a_whole_table_can_be_ruled_out():
    calls = [{"columns": ["Customer Geography[State]", "Sales[Freight Value]"]}]
    (problem,) = rules().violations(calls)
    assert problem.startswith("buyer state") and "'Customer Geography' (any column)" in problem
    assert rules().violations(calls + [{"columns": ["Customers[Customer State]", "Sales[Freight Value]"]}]) == []


def test_a_raw_column_ruled_out_in_favour_of_a_measure():
    (problem,) = rules().violations([{"columns": ["Sales[Price]", "Products[Product Category English]"]}])
    assert problem.startswith("item revenue") and "[Total Price]" in problem and "use the measure" in problem


def test_queries_that_touch_no_concept_are_ignored():
    assert rules().violations([{"columns": ["Date[Date]"], "measures": ["Total Orders"]}]) == []


def test_rules_load_from_a_dict_a_list_or_a_file(tmp_path):
    path = rules().save(tmp_path / "rules" / "sales.json")
    again = ColumnRules.load(path)
    assert again.to_dict() == rules().to_dict()
    assert ColumnRules.from_value(RULES["rules"]).rules[0].concept == "product category"
    with pytest.raises(ValueError, match="never"):
        ColumnRules.from_value([{"concept": "x", "use": ["A[B]"]}])
    with pytest.raises(FileNotFoundError):
        ColumnRules.from_value(tmp_path / "missing.json")


def test_a_semantic_model_takes_rules_and_names_them_after_itself(tmp_path):
    path = rules().save(tmp_path / "r.json")
    model = SemanticModel("Sales model", validate=False, rules=str(path))
    assert isinstance(model.rules, ColumnRules) and model.rules.model == "Sales model"
    text = repr(model.rules)
    assert "Rules for Sales model (3)" in text and "never Products[Product Category]" in text
    assert "Column rules for `model`" in model.rules.instructions("model")


def test_check_calls_with_several_models_judges_each_on_its_own_calls():
    a = SemanticModel("A", validate=False, rules=RULES)
    b = SemanticModel("B", validate=False)
    inputs = {"sales": a, "other": b}
    calls = [{**PORTUGUESE, "input": "other"}, {**ENGLISH, "input": "sales"}]
    assert check_calls(inputs, calls) == []
    assert len(check_calls({"sales": a, "more": SemanticModel("C", validate=False, rules=RULES)},
                           [{**PORTUGUESE, "input": "more"}])) == 1


class CatalogModel:
    dataset = "Sales model"

    def columns(self):
        return [{"Table Name": "Products", "Column Name": "Product Category"},
                {"Table Name": "Products", "Column Name": "Product Category English"},
                {"Table Name": "Customers", "Column Name": "Customer State"},
                {"Table Name": "Customer Geography", "Column Name": "State"}]

    def measures(self):
        return [{"Measure Name": "Total Price"}]


def test_plain_words_compile_into_rules_checked_against_the_model():
    seen = []

    def lm(*, messages):
        seen.append(messages[0]["content"])
        return json.dumps({"rules": [
            {"concept": "product category", "use": ["Products[Product Category English]"], "never": ["products[product category]"]},
            {"concept": "buyer state", "use": ["Customers[Customer State]"], "never": ["Customer Geography", "Customers[Zip]"]},
        ]})

    compiled = compile_rules(CatalogModel(), "Categories come from the English column. State is the customer's state.", lm)
    assert "Products[Product Category English]" in seen[0] and "[Total Price]" in seen[0]
    assert [r.never for r in compiled.rules] == [["Products[Product Category]"], ["Customer Geography"]]
    assert compiled.unresolved == ["buyer state: Customers[Zip]"]
    assert "Unresolved: buyer state: Customers[Zip]." in repr(compiled)


class ScriptedLM:
    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.messages = []

    def __call__(self, *, messages):
        self.messages.append(deepcopy(messages))
        return "```python\n" + self.blocks.pop(0) + "\n```"


def record(columns):
    return f"model._record_query({{'query_type': 'dax', 'executed': True, 'columns': {columns!r}, 'measures': ['Total Price']}})\n"


def test_a_run_that_computes_with_a_ruled_out_column_is_sent_back_and_then_verified():
    lm = ScriptedLM([
        record(["Products[Product Category English]"]) + record(["Products[Product Category]"]) + "SUBMIT(total=1.0)",
        record(["Products[Product Category English]"]) + "SUBMIT(total=2.0)",
    ])
    model = SemanticModel("Sales model", validate=False, rules=RULES)
    result = RLM.task("Total commission by category.", inputs={"model": model}, outputs={"total": float},
                      lm=lm, max_turns=4, timeout=60).run()
    assert result.submitted and result.payload == {"total": 2.0}
    assert "Column rules for `model`" in lm.messages[0][0]["content"]
    repair = lm.messages[1][-1]["content"]
    assert "semantic model column rules" in repair and "Products[Product Category English]" in repair
    assert result.verified
    assert result.trajectory.metadata["semantic_model_rules"]["violations"] == []


def test_without_rules_nothing_changes():
    lm = ScriptedLM([record(["Products[Product Category]"]) + "SUBMIT(total=1.0)"])
    result = RLM.task("Total.", inputs={"model": SemanticModel("Sales model", validate=False)}, outputs={"total": float},
                      lm=lm, max_turns=2, timeout=60).run()
    assert result.submitted and "semantic_model_rules" not in result.trajectory.metadata
    assert "Column rules" not in lm.messages[0][0]["content"]
