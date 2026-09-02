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
    """Enough DataFrame to satisfy schema(): columns, [], to_string, iterrows.

    Column names mirror what sempy actually returns, which matters: measures
    carry "Measure Description", not "Description", and getting that wrong
    drops the descriptions silently.
    """

    def __init__(self, rows, columns, label="<rows>"):
        self._rows = [dict(r) for r in rows]
        self.columns = list(columns)
        self._label = label

    def __getitem__(self, keep):
        if isinstance(keep, str):
            keep = [keep]
        return FakeFrame([{k: r.get(k) for k in keep} for r in self._rows],
                         keep, f"{self._label}|cols={list(keep)}")

    def iterrows(self):
        return enumerate(self._rows)

    def to_dict(self, orient="dict"):
        assert orient == "records"
        return [dict(row) for row in self._rows]

    def to_string(self):
        return f"{self._label} {self._rows}"


TABLES = [{"Name": "Sales", "Description": "fact"},
          {"Name": "Owner", "Description": ""}]
COLUMNS = [{"Table Name": "Sales", "Column Name": "Amount", "Description": ""},
           {"Table Name": "Sales", "Column Name": "Date", "Description": ""},
           {"Table Name": "Owner", "Column Name": "Country", "Description": "ISO"}]
MEASURES = [{"Table Name": "Sales", "Measure Name": "Total Sales",
             "Measure Expression": "SUM(Sales[Amount])",
             "Measure Description": "all channels",
             "Measure Display Folder": "Revenue"}]
RELS = [{"From Table": "Sales", "From Column": "OwnerId",
         "To Table": "Owner", "To Column": "Id", "Multiplicity": "m:1"}]


@pytest.fixture
def fake_sempy(monkeypatch):
    """Install a fake sempy.fabric and record every call made to it."""
    calls: list[tuple[str, tuple, dict]] = []

    def recorder(name, rows, columns):
        def fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return FakeFrame(rows, columns, f"<{name}>")
        return fn

    fabric = types.ModuleType("sempy.fabric")
    fabric.list_tables = recorder("list_tables", TABLES, list(TABLES[0]))
    fabric.list_columns = recorder("list_columns", COLUMNS, list(COLUMNS[0]))
    fabric.list_measures = recorder("list_measures", MEASURES, list(MEASURES[0]))
    fabric.list_relationships = recorder("list_relationships", RELS, list(RELS[0]))
    fabric.evaluate_dax = recorder(
        "evaluate_dax",
        [{
            "Period[Year]": 2024,
            "[ARR]": 100.0,
            "[ActiveCustomers]": 5,
        }],
        ["Period[Year]", "[ARR]", "[ActiveCustomers]"],
    )
    fabric.evaluate_measure = recorder("evaluate_measure", [{"v": 1}], ["v"])
    fabric.read_table = recorder("read_table", [{"v": 1}], ["v"])

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


def test_semantic_model_worker_wire_drops_parent_notebook_credential_provider():
    """The worker must use SemPy's established authentication path."""
    model = SemanticModel(
        "ARR Model SF (79)",
        workspace="Analytics",
        credential_provider="notebookutils",
        validate=False,
    )
    wire = encode_for_worker({"arr": model})

    import json
    json.dumps(wire), "the wire format must be JSON-safe"

    back = decode_from_worker_wire(wire)["arr"]
    assert isinstance(back, SemanticModel)
    assert back.dataset == "ARR Model SF (79)"
    assert back.workspace == "Analytics"
    assert back.credential_provider is None
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


def test_notebookutils_credential_is_threaded_through_every_call(fake_sempy):
    model = SemanticModel(
        "D",
        credential_provider="notebookutils",
        validate=False,
    )

    model.tables()
    model.measure("Total Sales")

    for _name, _args, kwargs in fake_sempy:
        credential = kwargs.get("credential")
        assert credential is not None
        assert callable(getattr(credential, "get_token", None))


def test_notebookutils_credential_fetches_pbi_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens: list[str] = []

    class AccessToken:
        def __init__(self, token: str, expires_on: int) -> None:
            self.token = token
            self.expires_on = expires_on

    class Credentials:
        @staticmethod
        def getToken(resource):
            tokens.append(resource)
            return (
                "eyJhbGciOiJub25lIn0."
                "eyJleHAiOjQxMDI5OTUyMDB9."
                "signature"
            )

    notebookutils = types.ModuleType("notebookutils")
    notebookutils.credentials = Credentials()
    azure = types.ModuleType("azure")
    azure_core = types.ModuleType("azure.core")
    azure_credentials = types.ModuleType("azure.core.credentials")
    azure_credentials.AccessToken = AccessToken
    azure.core = azure_core
    azure_core.credentials = azure_credentials
    monkeypatch.setitem(sys.modules, "notebookutils", notebookutils)
    monkeypatch.setitem(sys.modules, "azure", azure)
    monkeypatch.setitem(sys.modules, "azure.core", azure_core)
    monkeypatch.setitem(
        sys.modules,
        "azure.core.credentials",
        azure_credentials,
    )

    model = SemanticModel(
        "D",
        credential_provider="notebookutils",
        validate=False,
    )
    token = model._kw["credential"].get_token(
        "https://analysis.windows.net/powerbi/api/.default"
    )

    assert tokens == ["pbi"]
    assert token.token.startswith("eyJ")
    assert token.expires_on == 4102995200


def test_unknown_credential_provider_is_rejected():
    with pytest.raises(ValueError, match="credential_provider"):
        SemanticModel(
            "D",
            credential_provider="unknown",
            validate=False,
        )


def test_no_workspace_means_the_kwarg_is_absent(fake_sempy):
    """Passing workspace=None explicitly is not the same as omitting it."""
    SemanticModel("D", validate=False).tables()
    assert "workspace" not in fake_sempy[0][2]


def test_dax_passes_the_query_through(fake_sempy):
    SemanticModel("D", validate=False).dax("EVALUATE ROW(\"v\", 1)")
    name, args, _kw = fake_sempy[0]
    assert name == "evaluate_dax"
    assert args == ("D", 'EVALUATE ROW("v", 1)')


def test_metadata_returns_stable_plain_dataframes(fake_sempy):
    metadata = SemanticModel("D", validate=False).metadata()

    assert metadata.tables.columns.tolist() == [
        "table_name",
        "description",
    ]
    assert metadata.columns.columns.tolist() == [
        "table_name",
        "column_name",
        "description",
    ]
    assert metadata.measures.columns.tolist() == [
        "table_name",
        "measure_name",
        "measure_expression",
        "measure_description",
        "measure_display_folder",
    ]
    assert metadata.relationships.columns.tolist() == [
        "from_table",
        "from_column",
        "to_table",
        "to_column",
        "multiplicity",
    ]
    assert metadata.measures.iloc[0]["measure_name"] == "Total Sales"


def test_dax_can_return_plain_dataframe_with_normalized_columns(fake_sempy):
    result = SemanticModel("D", validate=False).dax(
        'EVALUATE ROW("v", 1)',
        normalize_columns=True,
    )

    assert result.columns.tolist() == [
        "period_year",
        "arr",
        "active_customers",
    ]
    assert result.iloc[0].to_dict() == {
        "period_year": 2024,
        "arr": 100.0,
        "active_customers": 5,
    }


def test_dax_normalization_disambiguates_colliding_columns(
    fake_sempy, monkeypatch
):
    import sempy.fabric as fabric

    monkeypatch.setattr(
        fabric,
        "evaluate_dax",
        lambda *args, **kwargs: FakeFrame(
            [{"[ARR]": 1, "ARR": 2}],
            ["[ARR]", "ARR"],
        ),
    )

    result = SemanticModel("D", validate=False).dax(
        "EVALUATE 1",
        normalize_columns=True,
    )

    assert result.columns.tolist() == ["arr", "arr_2"]


def test_dax_normalization_avoids_suffix_name_collisions(
    fake_sempy, monkeypatch
):
    import pandas as pd
    import sempy.fabric as fabric

    monkeypatch.setattr(
        fabric,
        "evaluate_dax",
        lambda *args, **kwargs: pd.DataFrame(
            [[1, 2, 3]],
            columns=["ARR", "ARR_2", "[ARR]"],
        ),
    )

    result = SemanticModel("D", validate=False).dax(
        "EVALUATE 1",
        normalize_columns=True,
    )

    assert result.columns.tolist() == ["arr", "arr_2", "arr_3"]
    assert result.iloc[0].tolist() == [1, 2, 3]


def test_dax_normalization_preserves_duplicate_source_columns(
    fake_sempy, monkeypatch
):
    import pandas as pd
    import sempy.fabric as fabric

    monkeypatch.setattr(
        fabric,
        "evaluate_dax",
        lambda *args, **kwargs: pd.DataFrame(
            [[1, 2]],
            columns=["[ARR]", "[ARR]"],
        ),
    )

    result = SemanticModel("D", validate=False).dax(
        "EVALUATE 1",
        normalize_columns=True,
    )

    assert result.columns.tolist() == ["arr", "arr_2"]
    assert result.iloc[0].tolist() == [1, 2]


def test_dax_normalization_preserves_empty_result_schema(
    fake_sempy, monkeypatch
):
    import pandas as pd
    import sempy.fabric as fabric

    monkeypatch.setattr(
        fabric,
        "evaluate_dax",
        lambda *args, **kwargs: pd.DataFrame(columns=["[ARR]", "Period[Year]"]),
    )

    result = SemanticModel("D", validate=False).dax(
        "EVALUATE 1",
        normalize_columns=True,
    )

    assert result.empty
    assert result.columns.tolist() == ["arr", "period_year"]


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
        "list_tables", "list_measures", "list_relationships", "list_columns"]


def test_schema_asks_for_measure_expressions_and_descriptions(fake_sempy):
    """Names misdescribe what measures compute; descriptions carry the meaning."""
    text = SemanticModel("Sales", validate=False).schema()
    assert "Measure Expression" in text and "Measure Description" in text


def test_schema_uses_sempys_actual_measure_description_column(fake_sempy):
    """sempy calls it "Measure Description". Asking for "Description" returns
    nothing and raises nothing, so this failure is invisible without a check."""
    import sempy.fabric as fabric

    real = fabric.list_measures

    def strict(*a, **kw):
        frame = real(*a, **kw)
        original = frame.__getitem__

        def only_real_columns(keep):
            missing = [c for c in keep if c not in frame.columns]
            assert not missing, f"asked for columns that do not exist: {missing}"
            return original(keep)

        frame.__getitem__ = only_real_columns
        return frame

    fabric.list_measures = strict
    try:
        SemanticModel("Sales", validate=False).schema()
    finally:
        fabric.list_measures = real


def test_schema_lists_columns_grouped_by_table(fake_sempy):
    """Omitting columns cost 16 separate .columns() calls in one run."""
    text = SemanticModel("Sales", validate=False).schema()
    assert "== Columns ==" in text
    assert "list_columns" in [n for n, _a, _k in fake_sempy]


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
