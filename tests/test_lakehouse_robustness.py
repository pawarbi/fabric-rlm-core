"""Three ways a lakehouse task went wrong in Fabric without anything raising usefully.

* A failed Delta write left ``Tables/dbo/sales/_delta_log/`` empty. Discovery
  of the whole lakehouse then failed, for every table, and the error did not
  say which folder was at fault.
* Tables bound as separate handles cannot be joined. The error gave no way
  out, and three runs pulled whole tables to join them in Python.
* Those pulls came back capped at 1,000 rows with ``truncated: True``. Nothing
  printed, the generated code read ``rows`` only, and a run submitted a 0.00%
  lapse rate computed from 1,000 of 378,791 rows with ``integrity_ok`` true.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fabric_rlm import LakehouseSource
from fabric_rlm import lakehouse as lakehouse_module
from fabric_rlm.analytical_integrity import check_truncated_source_reads
from fabric_rlm.artifacts import decode_from_worker_wire, encode_for_worker
from fabric_rlm.lakehouse import LakehouseDiscoveryError, execute_lakehouse_query

ROOT = "abfss://ws@onelake.dfs.fabric.microsoft.com/lh"


class _Item:
    def __init__(self, path: str, *, is_dir: bool = True):
        self.path = path
        self.name = path.rstrip("/").rsplit("/", 1)[-1]
        self.isDir = is_dir


class _FakeFS:
    def __init__(self, listings):
        self.listings = listings

    def ls(self, path):
        if path not in self.listings:
            raise FileNotFoundError(path)
        return self.listings[path]


def _lakehouse(monkeypatch, tables: dict[str, bool]):
    """``tables`` maps a table name under Tables/dbo to whether it is readable."""

    dbo = f"{ROOT}/Tables/dbo"
    listings = {f"{ROOT}/Tables": [_Item(dbo)], dbo: [_Item(f"{dbo}/{t}") for t in tables]}
    for name in tables:
        listings[f"{dbo}/{name}"] = [_Item(f"{dbo}/{name}/_delta_log")]

    def read(path):
        if not tables[path.rsplit("/", 1)[-1]]:
            raise LakehouseDiscoveryError(
                "Delta transaction metadata could not be read. Delta-RS reader "
                "failed with TableNotFoundError."
            )
        return {"columns": [["id", "BIGINT"]], "table_id": f"id:{path}", "version": 1}

    monkeypatch.setattr("fabric_rlm.lakehouse._get_fs", lambda: _FakeFS(listings))
    monkeypatch.setattr("fabric_rlm.lakehouse._read_delta_metadata", read)


# -- unreadable table folders ---------------------------------------------------


def test_one_unreadable_table_does_not_hide_the_others(monkeypatch) -> None:
    _lakehouse(monkeypatch, {"sales": True, "half_written": False, "customers": True})

    resolved = LakehouseSource(ROOT).resolve()

    assert [entry["name"] for entry in resolved.catalog] == ["dbo.customers", "dbo.sales"]
    assert [item["name"] for item in resolved.skipped] == ["dbo.half_written"]
    assert resolved.skipped[0]["path"].endswith("/Tables/dbo/half_written")
    assert "TableNotFoundError" in resolved.skipped[0]["reason"]


def test_the_model_is_told_what_could_not_be_read(monkeypatch) -> None:
    _lakehouse(monkeypatch, {"sales": True, "half_written": False})
    description = LakehouseSource(ROOT).resolve().__rlm_describe__()
    assert "1 table folder could not be read" in description
    assert "dbo.half_written" in description


def test_a_clean_lakehouse_describes_itself_without_the_notice(monkeypatch) -> None:
    _lakehouse(monkeypatch, {"sales": True})
    resolved = LakehouseSource(ROOT).resolve()
    assert resolved.skipped == ()
    assert "could not be read" not in resolved.__rlm_describe__()
    assert "skipped" not in resolved.__frozen__()


def test_a_scope_that_is_the_unreadable_table_still_fails_and_names_it(monkeypatch) -> None:
    _lakehouse(monkeypatch, {"sales": True, "half_written": False})
    with pytest.raises(LakehouseDiscoveryError, match="Tables/dbo/half_written"):
        LakehouseSource(f"{ROOT}/Tables/dbo/half_written").resolve()


def test_nothing_readable_at_all_fails_and_lists_the_folders(monkeypatch) -> None:
    _lakehouse(monkeypatch, {"a": False, "b": False})
    with pytest.raises(LakehouseDiscoveryError, match="could read none of them.*dbo.a"):
        LakehouseSource(ROOT).resolve()


def test_skipped_folders_survive_the_trip_to_the_worker(monkeypatch) -> None:
    _lakehouse(monkeypatch, {"sales": True, "half_written": False})
    resolved = LakehouseSource(ROOT).resolve()

    decoded = decode_from_worker_wire(encode_for_worker(resolved))

    assert decoded.skipped == resolved.skipped
    assert decoded == resolved
    # A handle encoded by an older parent has no "skipped" key.
    legacy = encode_for_worker(resolved)
    legacy["__fabric_rlm_lakehouse_source__"].pop("skipped")
    assert decode_from_worker_wire(legacy).skipped == ()


def test_querying_an_unreadable_table_says_why(monkeypatch) -> None:
    _lakehouse(monkeypatch, {"sales": True, "half_written": False})
    resolved = LakehouseSource(ROOT).resolve()
    with pytest.raises(ValueError, match="found but could not be read.*TableNotFoundError"):
        execute_lakehouse_query(resolved, sql="SELECT 1 FROM t", sources={"t": "dbo.half_written"})


# -- a table that lives on another handle -----------------------------------------


def test_the_not_in_catalog_error_says_how_to_join_tables() -> None:
    fact = LakehouseSource(
        ROOT, catalog=[{"kind": "delta", "name": "fact_policy_month", "path": f"{ROOT}/Tables/f"}]
    )
    with pytest.raises(ValueError) as raised:
        execute_lakehouse_query(
            fact,
            sql="SELECT 1 FROM f JOIN p USING (product_key)",
            sources={"f": "fact_policy_month", "p": "dim_product"},
        )
    message = str(raised.value)
    assert "'dim_product' is not in this LakehouseSource catalog" in message
    assert "This handle holds: fact_policy_month" in message
    assert 'tables=["Tables/<a>", "Tables/<b>"]' in message
    assert "max_rows" in message


# -- truncation -----------------------------------------------------------------


def test_a_truncated_result_is_announced_inside_a_worker(monkeypatch, capsys) -> None:
    source = LakehouseSource(ROOT, catalog=[{"kind": "delta", "name": "t", "path": f"{ROOT}/Tables/t"}])
    monkeypatch.setattr(
        lakehouse_module,
        "_HOST_QUERY_TRANSPORT",
        lambda **kwargs: {"columns": ["a"], "rows": [[1]] * 1000, "truncated": True},
    )

    result = source.query("SELECT a FROM t", sources={"t": "t"})

    assert result["truncated"] is True and len(result["rows"]) == 1000
    printed = capsys.readouterr().out
    assert "truncated this result at 1,000 rows" in printed
    assert "Aggregate in SQL" in printed and "10,000" in printed


def test_a_complete_result_prints_nothing(monkeypatch, capsys) -> None:
    source = LakehouseSource(ROOT, catalog=[{"kind": "delta", "name": "t", "path": f"{ROOT}/Tables/t"}])
    monkeypatch.setattr(
        lakehouse_module,
        "_HOST_QUERY_TRANSPORT",
        lambda **kwargs: {"columns": ["n"], "rows": [[378791]], "truncated": False},
    )
    source.query("SELECT count(*) n FROM t", sources={"t": "t"})
    assert capsys.readouterr().out == ""


def _turn(number, *calls):
    return SimpleNamespace(turn=number, source_calls=list(calls))


def _call(truncated, root=ROOT, **extra):
    return {"query_type": "lakehouse_sql", "source_root": root, "truncated": truncated,
            "max_rows": 1000, **extra}


def test_a_capped_pull_that_nothing_replaced_is_an_integrity_problem() -> None:
    problems = check_truncated_source_reads([_turn(1, _call(False)), _turn(2, _call(True))])
    assert len(problems) == 1
    assert "Turn 2" in problems[0] and "max_rows=1000" in problems[0] and "aggregate in SQL" in problems[0]


def test_a_preview_followed_by_a_real_aggregate_is_fine() -> None:
    assert check_truncated_source_reads([_turn(1, _call(True)), _turn(2, _call(False))]) == []


def test_sources_are_judged_separately_and_failures_are_ignored() -> None:
    other = "abfss://ws@onelake.dfs.fabric.microsoft.com/other"
    turns = [
        _turn(1, _call(True, root=other)),
        _turn(2, _call(False), _call(True, reason="execution_error")),
        _turn(3, {"query_type": "aggregate", "truncated": True}),      # a semantic-model call
    ]
    problems = check_truncated_source_reads(turns)
    assert len(problems) == 1 and "Turn 1" in problems[0]


def test_turns_without_telemetry_are_tolerated() -> None:
    assert check_truncated_source_reads([SimpleNamespace(turn=1), _turn(2)]) == []
