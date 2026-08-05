"""SemanticModel as a bound input.

A task that names a semantic model but gives no way in scores 7/19 and 5/15 on
two different models, with most questions burning every turn. The fix is to
make the connection exist rather than describe it, so these tests care about
two things: the handle survives the trip to the worker, and it introduces
itself in the prompt. Everything else is plumbing around those.

sempy is faked here so this runs in CI, which has no Fabric.
"""

from __future__ import annotations

import sys
import types

import pytest

from fabric_rlm import SemanticModel
from fabric_rlm.artifacts import decode_from_worker_wire, encode_for_worker
from fabric_rlm.prompts import _describe_value


class FakeFrame:
    """Enough DataFrame to satisfy schema()."""

    def __init__(self, columns, text="<rows>"):
        self.columns = list(columns)
        self._text = text

    def __getitem__(self, keep):
        return FakeFrame(keep, f"{self._text}|cols={keep}")

    def to_string(self):
        return self._text


@pytest.fixture
def fake_sempy(monkeypatch):
    """Install a fake sempy.fabric and record every call made to it."""
    calls: list[tuple[str, tuple, dict]] = []

    def recorder(name, columns=("A",)):
        def fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return FakeFrame(columns, f"<{name}>")
        return fn

    fabric = types.ModuleType("sempy.fabric")
    fabric.list_tables = recorder("list_tables", ("Name", "Description"))
    fabric.list_columns = recorder("list_columns")
    fabric.list_measures = recorder(
        "list_measures", ("Table Name", "Measure Name", "Measure Expression", "Description"))
    fabric.list_relationships = recorder(
        "list_relationships", ("From Table", "To Table"))
    fabric.evaluate_dax = recorder("evaluate_dax")
    fabric.evaluate_measure = recorder("evaluate_measure")
    fabric.read_table = recorder("read_table")

    sempy = types.ModuleType("sempy")
    sempy.fabric = fabric
    monkeypatch.setitem(sys.modules, "sempy", sempy)
    monkeypatch.setitem(sys.modules, "sempy.fabric", fabric)
    return calls


@pytest.fixture
def no_sempy(monkeypatch):
    """Make `import sempy.fabric` fail, as it does outside Fabric."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def fake_import(name, *a, **kw):
        if name == "sempy" or name.startswith("sempy."):
            raise ImportError("no sempy here")
        return real_import(name, *a, **kw)

    monkeypatch.delitem(sys.modules, "sempy", raising=False)
    monkeypatch.delitem(sys.modules, "sempy.fabric", raising=False)
    monkeypatch.setattr("builtins.__import__", fake_import)


# -- the two things that matter -------------------------------------------


def test_survives_the_trip_to_the_worker():
    """The parent holds a validated handle; the worker must rebuild a usable one."""
    model = SemanticModel("ARR Model SF (79)", workspace="Analytics", validate=False)
    wire = encode_for_worker({"arr": model})

    import json
    json.dumps(wire), "the wire format must be JSON-safe"

    back = decode_from_worker_wire(wire)["arr"]
    assert isinstance(back, SemanticModel)
    assert back.dataset == "ARR Model SF (79)"
    assert back.workspace == "Analytics"
    assert back == model


def test_the_worker_side_handle_does_not_revalidate(no_sempy):
    """Decoding runs in the worker with no network. It must not call out."""
    wire = {"__fabric_rlm_semantic_model__": {"dataset": "D", "workspace": None}}
    assert decode_from_worker_wire(wire).dataset == "D"


def test_it_introduces_itself_in_the_prompt():
    """If the listing does not name the methods, the model does not know they exist."""
    line = _describe_value(SemanticModel("Sales", validate=False))
    assert "SemanticModel" in line and "'Sales'" in line
    for hook in (".schema()", ".dax(", ".measure("):
        assert hook in line, f"{hook} missing from the input listing"
    assert len(line) < 400, "the point of binding a handle is not paying for prose"


# -- validation ------------------------------------------------------------


def test_auto_validation_is_skipped_without_sempy(no_sempy):
    """Constructible off-Fabric, so tests and dry runs work."""
    assert SemanticModel("Sales").dataset == "Sales"


def test_explicit_validation_without_sempy_is_a_clear_error(no_sempy):
    with pytest.raises(RuntimeError) as err:
        SemanticModel("Sales", validate=True)
    msg = str(err.value)
    assert "sempy" in msg
    assert "import fabric" in msg, "must head off the SSH-package confusion"


def test_auto_validation_runs_when_sempy_is_present(fake_sempy):
    SemanticModel("Sales", workspace="WS")
    assert fake_sempy and fake_sempy[0][0] == "list_tables"
    assert fake_sempy[0][2] == {"workspace": "WS"}


def test_unreachable_model_names_the_dataset(fake_sempy, monkeypatch):
    import sempy.fabric as fabric

    def boom(*a, **kw):
        raise KeyError("not found")

    monkeypatch.setattr(fabric, "list_tables", boom)
    with pytest.raises(ValueError, match="Typo Model"):
        SemanticModel("Typo Model", validate=True)


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError):
        SemanticModel("   ", validate=False)


# -- querying --------------------------------------------------------------


def test_workspace_is_threaded_through_every_call(fake_sempy):
    m = SemanticModel("D", workspace="WS", validate=False)
    m.tables(); m.columns(); m.measures(); m.relationships()
    m.dax("EVALUATE 1")
    m.read_table("Dim", num_rows=10)
    assert all(kw.get("workspace") == "WS" for _n, _a, kw in fake_sempy)


def test_no_workspace_means_the_kwarg_is_absent(fake_sempy):
    """Passing workspace=None explicitly is not the same as omitting it."""
    SemanticModel("D", validate=False).tables()
    assert "workspace" not in fake_sempy[0][2]


def test_dax_passes_the_query_through(fake_sempy):
    SemanticModel("D", validate=False).dax("EVALUATE ROW(\"v\", 1)")
    name, args, _kw = fake_sempy[0]
    assert name == "evaluate_dax"
    assert args == ("D", 'EVALUATE ROW("v", 1)')


def test_measure_maps_to_sempy_argument_names(fake_sempy):
    SemanticModel("D", validate=False).measure(
        "Total Sales", groupby=["Owner[Country]"], filters={"Owner[Tier]": ["Tier1"]})
    name, args, kw = fake_sempy[0]
    assert name == "evaluate_measure" and args == ("D", "Total Sales")
    assert kw["groupby_columns"] == ["Owner[Country]"]
    assert kw["filters"] == {"Owner[Tier]": ["Tier1"]}


def test_measure_omits_empty_grouping(fake_sempy):
    SemanticModel("D", validate=False).measure("Total Sales")
    _n, _a, kw = fake_sempy[0]
    assert "groupby_columns" not in kw and "filters" not in kw


def test_read_table_omits_num_rows_when_unset(fake_sempy):
    SemanticModel("D", validate=False).read_table("Dim")
    assert "num_rows" not in fake_sempy[0][2]


# -- schema ----------------------------------------------------------------


def test_schema_is_one_call_covering_the_first_turn(fake_sempy):
    text = SemanticModel("Sales", validate=False).schema()
    assert "Sales" in text
    for heading in ("== Tables ==", "== Measures ==", "== Relationships =="):
        assert heading in text
    assert [n for n, _a, _k in fake_sempy] == [
        "list_tables", "list_measures", "list_relationships"]


def test_schema_asks_for_measure_expressions_and_descriptions(fake_sempy):
    """Names misdescribe what measures compute; descriptions carry the meaning."""
    text = SemanticModel("Sales", validate=False).schema()
    assert "Measure Expression" in text and "Description" in text


def test_schema_survives_one_section_failing(fake_sempy, monkeypatch):
    """A model with no relationships must not lose its tables and measures."""
    import sempy.fabric as fabric

    def boom(*a, **kw):
        raise ValueError("nope")

    monkeypatch.setattr(fabric, "list_relationships", boom)
    text = SemanticModel("Sales", validate=False).schema()
    assert "== Tables ==" in text
    assert "unavailable" in text and "ValueError" in text


# -- several models at once ------------------------------------------------


def test_two_models_bind_independently():
    inputs = {
        "sales": SemanticModel("Sales Model", workspace="WS-A", validate=False),
        "ops": SemanticModel("Ops Model", workspace="WS-B", validate=False),
    }
    back = decode_from_worker_wire(encode_for_worker(inputs))
    assert back["sales"].dataset == "Sales Model"
    assert back["ops"].dataset == "Ops Model"
    assert back["sales"].workspace == "WS-A"
    assert back["ops"].workspace == "WS-B"
    assert back["sales"] != back["ops"]


def test_two_models_are_listed_separately_in_the_prompt():
    from fabric_rlm.prompts import build_system_prompt

    prompt = build_system_prompt(
        inline_task="compare them",
        inputs={
            "sales": SemanticModel("Sales Model", validate=False),
            "ops": SemanticModel("Ops Model", validate=False),
        },
        inline_outputs=["answer"],
    )
    assert "'Sales Model'" in prompt and "'Ops Model'" in prompt
    assert prompt.count("SemanticModel dataset=") == 2


def test_queries_on_two_models_do_not_cross(fake_sempy):
    a = SemanticModel("A", workspace="WS-A", validate=False)
    b = SemanticModel("B", workspace="WS-B", validate=False)
    a.dax("EVALUATE 1")
    b.dax("EVALUATE 2")
    assert fake_sempy[0][1] == ("A", "EVALUATE 1")
    assert fake_sempy[0][2]["workspace"] == "WS-A"
    assert fake_sempy[1][1] == ("B", "EVALUATE 2")
    assert fake_sempy[1][2]["workspace"] == "WS-B"
